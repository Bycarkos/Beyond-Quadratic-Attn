"""Evaluate a trained SSOG or dense-MHA checkpoint at patch sizes it was not
trained on.

The image resolution is kept fixed, so a smaller patch gives a finer token
grid (224px: p16 -> 14x14, p14 -> 16x16, p8 -> 28x28). Trained weights are
carried over with:

* patch-embedding kernel: FlexiViT PI-resize (least-squares preserving
  <x, w> = <resize(x), w'>), not a plain bilinear resize;
* position embeddings: bicubic interpolation onto the new grid;
* SSOG geometry (dense MHA has none, so it only uses "token"), in one of two modes:
    token -- mu/sigma kept in token units (the kernel covers less image area);
    image -- mu, sigma, sigma_floor and max_offset scaled by new/old grid, so
             every Gaussian covers the same image area as in training.

Usage: python eval_patch_size.py -c evaluation/patch_size/cifar100/<model>.yaml
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from dataset.collator import Collator
from models.DiscreteSSOGViT import DiscreteSSOG_ViT
from models.MatchedViT import MatchedBViT, MatchedViT
from models.ODEViT import ODE_ViT
from models.SSOGViT import SSOG_ViT, CenterNorm, SSOGAttention, _EPS

MODEL_CLASSES = {
    "continuous": SSOG_ViT, "discrete": DiscreteSSOG_ViT, "looped": DiscreteSSOG_ViT,
    "dense_continuous": ODE_ViT, "dense_discrete": MatchedViT, "dense_looped": MatchedBViT,
}
# Patch-embedding kernel / position-embedding keys: SSOG + ODE_ViT, torchvision ViT.
KERNEL_KEYS = ("patch_embeddings.patch_embed.proj.weight", "vit.conv_proj.weight")
POS_KEYS = ("patch_embeddings.pos_embed", "vit.encoder.pos_embedding")

torch.set_float32_matmul_precision("medium")


def resolve_checkpoint(cfg):
    choice = str(cfg.evaluation.checkpoint)
    if choice not in {"best", "last"}:
        return Path(choice).expanduser()
    last = Path(str(cfg.modeling.checkpoint_path)).expanduser() / "last.ckpt"
    if choice == "last":
        return last
    callbacks = torch.load(last, map_location="cpu", weights_only=False)["callbacks"]
    best = next(v["best_model_path"] for k, v in callbacks.items() if "ModelCheckpoint" in k)
    return Path(best)


def pi_resize_kernel(weight, new_size):
    old_size = weight.shape[-1]
    if new_size == old_size:
        return weight
    basis = torch.eye(old_size ** 2, dtype=torch.float64).reshape(-1, 1, old_size, old_size)
    resize = F.interpolate(basis, size=(new_size, new_size), mode="bilinear",
                           align_corners=False, antialias=True).reshape(old_size ** 2, -1)
    flat = weight.double().reshape(-1, old_size ** 2).T
    return (torch.linalg.pinv(resize) @ flat).T.reshape(
        *weight.shape[:2], new_size, new_size).to(weight.dtype)


def resize_pos_embed(pos_embed, old_grid, new_grid):
    num_prefix = pos_embed.shape[1] - old_grid ** 2  # CLS token(s), if any
    prefix, grid = pos_embed[:, :num_prefix], pos_embed[:, num_prefix:]
    grid = grid.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return torch.cat([prefix, grid.permute(0, 2, 3, 1).reshape(1, new_grid ** 2, -1)], dim=1)


@torch.no_grad()
def rescale_geometry(model, scale):
    for attn in model.modules():
        if not isinstance(attn, SSOGAttention):
            continue
        attn.mu.mul_(scale)
        # sigma = softplus(raw) + EPS + floor; target sigma' = scale * sigma
        # with floor' = scale * floor, so softplus(raw') = scale*(softplus(raw)+EPS) - EPS.
        target = scale * (F.softplus(attn.raw_sigma) + _EPS) - _EPS
        attn.raw_sigma.copy_(target + torch.log(-torch.expm1(-target)))
        attn.sigma_floor *= scale
        attn.max_offset *= scale


def match_legacy_layout(model, state, dim, num_classes):
    """Checkpoints trained before the LayerNorm/GELU switch (commit 47e0504)
    used CenterNorm ([1, D] affine), ELU MLPs and, for discrete, a
    LayerNorm+Linear head on the pooled tokens instead of final_norm."""
    if isinstance(model, SSOG_ViT) and state["equation.norm_attn.weight"].ndim == 2:
        model.equation.norm_attn = CenterNorm(dim)
        model.equation.norm_mlp = CenterNorm(dim)
    if isinstance(model, DiscreteSSOG_ViT) and "classifier.0.weight" in state:
        for block in model.blocks:
            block.norm_attn, block.norm_mlp = CenterNorm(dim), CenterNorm(dim)
            block.mlp.act = nn.ELU()
        model.final_norm = nn.Identity()
        model.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))


def build_model(cfg, state_dict, patch_size, geometry):
    inputs = dict(cfg.modeling.inputs)
    old_grid = inputs["img_size"] // inputs["patch_size"]
    new_grid = inputs["img_size"] // patch_size
    inputs["patch_size"] = patch_size
    model = MODEL_CLASSES[cfg.evaluation.model](**inputs)

    state = {k.removeprefix("model."): v for k, v in state_dict.items()}
    match_legacy_layout(model, state, inputs["dim"], inputs["num_classes"])
    kernel_key = next(k for k in KERNEL_KEYS if k in state)
    pos_key = next(k for k in POS_KEYS if k in state)
    state[kernel_key] = pi_resize_kernel(state[kernel_key], patch_size)
    state[pos_key] = resize_pos_embed(state[pos_key], old_grid, new_grid)
    model.load_state_dict(state, strict=True)

    if geometry == "image" and new_grid != old_grid:
        rescale_geometry(model, new_grid / old_grid)
    return model


def build_loader(cfg):
    size = cfg.modeling.inputs.img_size
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    name, root = cfg.data.dataset.name, cfg.data.dataset.dataset_path
    if name in {"cifar100", "cifar10"}:
        dataset = (CIFAR100 if name == "cifar100" else CIFAR10)(root=root, train=False, download=False)
    else:
        from datasets import load_dataset
        dataset = load_dataset(name, cache_dir=root)["validation"]
    collator = Collator(processor=None, use_processor=False, transform=transform, train=False)
    return DataLoader(dataset, batch_size=cfg.evaluation.batch_size, shuffle=False,
                      num_workers=cfg.evaluation.num_workers, pin_memory=True,
                      collate_fn=collator.classification_collate_fn)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    correct1 = correct5 = total = 0
    for batch in loader:
        images = batch["pixel_values"]["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        top5 = model(images).logits.topk(5, dim=-1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=-1).sum().item()
        total += labels.numel()
    return 100 * correct1 / total, 100 * correct5 / total


def main(cfg):
    device = torch.device("cuda")
    checkpoint = resolve_checkpoint(cfg)
    print(f"Checkpoint: {checkpoint}", flush=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    loader = build_loader(cfg)

    rows = []
    for patch_size in cfg.evaluation.patch_sizes:
        for geometry in cfg.evaluation.geometry:
            if patch_size == cfg.modeling.inputs.patch_size and geometry != cfg.evaluation.geometry[0]:
                continue  # both modes are identical at the training patch size
            model = build_model(cfg, state_dict, patch_size, geometry)
            acc1, acc5 = evaluate(model, loader, device)
            grid = cfg.modeling.inputs.img_size // patch_size
            rows.append(dict(model=cfg.evaluation.model, patch_size=patch_size, grid=grid,
                             geometry=geometry, acc_1=round(acc1, 2), acc_5=round(acc5, 2)))
            print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache()

    output = Path(cfg.evaluation.output_dir) / f"{cfg.evaluation.model}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True,
                        help="Evaluation YAML file from configs/classification.")
    args = parser.parse_args()
    with initialize(version_base=None, config_path="configs/classification"):
        main(compose(config_name=args.config_file))
