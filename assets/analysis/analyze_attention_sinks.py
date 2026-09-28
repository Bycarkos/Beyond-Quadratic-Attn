#!/usr/bin/env python3
"""Attention-sink analysis: trained dense MHA vs SSoG models (read-only).

Loads existing checkpoints through eval_patch_size.build_model (patch 16,
i.e. exactly the trained models) and measures, on validation images:

  sink score    N * max_j mean_i A_ij per head: the most-attended key's
                received mass relative to uniform (1 = none, >> 1 = sink)
  consistency   fraction of images whose most-attended key is the head's
                most common one (a content-independent sink scores ~1)
  cls mass      mean attention from patch queries to the CLS token
  entropy       mean query entropy / log N
  rollout sink  sink score of the residual attention rollout
                prod_l (A_l + I) / 2 over layers/steps (head-averaged)
  high-norm     fraction of final pre-norm patch tokens with norm > 3x the
                image median, and the max/median ratio (Darcet et al., 2024)

Dense attention is recomputed from each nn.MultiheadAttention's own weights
with a read-only forward pre-hook (softmax(QK^T / sqrt(d))); SSoG attention
is its own dense-equivalent matrix, obtained by setting the existing
capture_attention flag on the loaded module objects. No file is modified.

python -m assets.analysis.analyze_attention_sinks [--num-images 512] [--device cuda]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import open_dict
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import eval_patch_size as E
from models.DiscreteSSOGViT import DiscreteSSOG_ViT
from models.MatchedViT import MatchedBViT, MatchedViT
from models.SSOGViT import SSOGAttention

OUTPUT_DIR = Path("experiments/SSOG/attention_sinks")
CKPT = Path("~/checkpoints/SSOG-Paper").expanduser()

# (label, dataset, operator, config, evaluation.model, checkpoint or None = best)
MODELS = [
    ("Continuous", "cifar100", "MHA", "evaluation/patch_size/cifar100/dense_continuous.yaml", "dense_continuous", None),
    ("Continuous", "cifar100", "SSoG", "evaluation/patch_size/cifar100/continuous.yaml", "continuous", None),
    ("Discrete", "cifar100", "MHA", "evaluation/patch_size/cifar100/dense_discrete.yaml", "dense_discrete", None),
    ("Discrete", "cifar100", "SSoG", "evaluation/patch_size/cifar100/discrete.yaml", "discrete", None),
    ("Looped", "cifar100", "MHA", "evaluation/patch_size/cifar100/dense_looped.yaml", "dense_looped", None),
    ("Looped", "cifar100", "SSoG", "evaluation/patch_size/cifar100/looped.yaml", "looped", None),
    ("Discrete", "cifar100", "SSoG steered", "ssog/discrete/cifar100/steered_dim768_h12_r4_depth12.yaml", "discrete", None),
    ("Looped", "cifar100", "SSoG steered", "ssog/looped_ssog/cifar100/steered_dim768_h12_r4_depth12.yaml", "looped", None),
    ("Discrete", "imagenet", "MHA", "ssog/baselines/imagenet/vit_b16_resume_raw_lr2e4.yaml", "dense_discrete",
     CKPT / "imagenet1k-matched-baselines/imagenet-vit-b16-depth12-raw-resume-lr2e4-seed42/"
            "286-imagenet-vit-b16-depth12-raw-resume-lr2e4-seed42.pt.ckpt"),
    ("Discrete", "imagenet", "SSoG", "evaluation/patch_size/imagenet/discrete.yaml", "discrete", None),
]


def load_config(name, model_key, checkpoint):
    with initialize(version_base=None, config_path="../../configs/classification"):  # relative to this file
        cfg = compose(config_name=name)
    with open_dict(cfg):  # in-memory only: the files are not touched
        cfg.setdefault("evaluation", {})
        cfg.evaluation.model = model_key
        cfg.evaluation.checkpoint = str(checkpoint) if checkpoint else cfg.evaluation.get("checkpoint", "best")
        cfg.evaluation.setdefault("batch_size", 32)
        cfg.evaluation.setdefault("num_workers", 4)
    return cfg


def imagenet_loader(num_images, batch_size):
    from datasets import Dataset
    from tests.ssog_experiments.common import find_imagenet_arrow_shards
    dataset = Dataset.from_file(str(find_imagenet_arrow_shards()[0]))
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def collate(items):
        return (torch.stack([transform(x["image"].convert("RGB")) for x in items]),
                torch.tensor([x["label"] for x in items]))
    return DataLoader(Subset(dataset, range(num_images)), batch_size=batch_size, collate_fn=collate)


def cifar_loader(cfg, num_images, batch_size):
    base = E.build_loader(cfg)

    def collate(items):
        batch = base.collate_fn(items)
        return batch["pixel_values"]["pixel_values"], batch["labels"]
    return DataLoader(Subset(base.dataset, range(num_images)), batch_size=batch_size, collate_fn=collate)


class AttentionRecorder:
    """Collects [B, H, N, N] attention of every attention call, in call order."""

    def __init__(self, model):
        self.maps, self.handles = [], []
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                self.handles.append(module.register_forward_pre_hook(self._mha, with_kwargs=True))
            elif isinstance(module, SSOGAttention):
                module.capture_attention = True  # existing runtime flag on the loaded object
                self.handles.append(module.register_forward_hook(self._ssog))

    def _mha(self, module, args, kwargs):
        x = args[0] if args else kwargs["query"]
        if not module.batch_first:
            x = x.transpose(0, 1)
        d, h = module.embed_dim, module.num_heads
        w, b = module.in_proj_weight, module.in_proj_bias
        q = x @ w[:d].T + (b[:d] if b is not None else 0)
        k = x @ w[d:2 * d].T + (b[d:2 * d] if b is not None else 0)
        bsz, n, _ = x.shape
        q = q.view(bsz, n, h, d // h).transpose(1, 2)
        k = k.view(bsz, n, h, d // h).transpose(1, 2)
        self.maps.append(torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // h), dim=-1).float())

    def _ssog(self, module, args, output):
        self.maps.append(output[1].float())

    def pop(self):
        maps, self.maps = self.maps, []
        return maps


def final_patch_tokens(model, images):
    """Pre-norm token states after the last block / step, and whether index 0 is CLS."""
    if isinstance(model, (MatchedViT, MatchedBViT, DiscreteSSOG_ViT)):
        out = model(images, output_hidden_states=True)
    else:
        out = model(images)
    tokens = out.hidden_states[-1]
    final_patch_tokens.logits = out.logits
    has_cls = isinstance(model, (MatchedViT, MatchedBViT)) or tokens.shape[1] == int(math.isqrt(tokens.shape[1] - 1)) ** 2 + 1
    return tokens, has_cls


def sink_stats(a):
    """a: [B, H, N, N] -> per-image-head sink score and argmax key."""
    received = a.mean(dim=-2)                      # mean over queries -> [B, H, N]
    top = received.max(dim=-1)
    return (a.shape[-1] * top.values), top.indices


@torch.no_grad()
def analyse(label, dataset, operator, cfg, loader, device):
    state = torch.load(E.resolve_checkpoint(cfg), map_location="cpu", weights_only=False)["state_dict"]
    model = E.build_model(cfg, state, int(cfg.modeling.inputs.patch_size), "token").eval().to(device)
    recorder = AttentionRecorder(model)

    per_layer = {}   # layer -> lists
    rollout_sink, cls_mass, norm_frac, norm_ratio = [], [], [], []
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        tokens, has_cls = final_patch_tokens(model, images)
        correct += (final_patch_tokens.logits.argmax(-1).cpu() == labels).sum().item()
        total += labels.numel()
        maps = recorder.pop()
        n = maps[0].shape[-1]
        rollout = torch.eye(n, device=device).expand(images.shape[0], n, n)
        for layer, a in enumerate(maps):
            s, idx = sink_stats(a)
            ent = -(a.clamp_min(1e-12).log() * a).sum(-1).mean(-1) / math.log(n)  # [B, H]
            rec = per_layer.setdefault(layer, {"sink": [], "idx": [], "entropy": [], "cls": []})
            rec["sink"].append(s.cpu()); rec["idx"].append(idx.cpu()); rec["entropy"].append(ent.cpu())
            if has_cls and n == tokens.shape[1]:
                rec["cls"].append(a[:, :, 1:, 0].mean(dim=(-1, -2)).cpu())  # patches -> CLS
            rollout = (0.5 * a.mean(dim=1) + 0.5 * torch.eye(n, device=device)) @ rollout
        rollout_sink.append((n * rollout.mean(dim=-2).max(dim=-1).values).cpu())
        patches = tokens[:, 1:] if has_cls else tokens
        norms = patches.norm(dim=-1)
        median = norms.median(dim=-1, keepdim=True).values
        norm_frac.append((norms > 3 * median).float().mean(-1).cpu())
        norm_ratio.append((norms.max(dim=-1).values / median.squeeze(-1)).cpu())

    rows = []
    for layer, rec in sorted(per_layer.items()):
        sink = torch.cat(rec["sink"]); idx = torch.cat(rec["idx"])
        mode = torch.mode(idx, dim=0).values                       # per head
        consistency = (idx == mode).float().mean(0)                # per head
        rows.append(dict(setting=label, dataset=dataset, operator=operator, layer=layer + 1,
                         sink_mean=sink.mean().item(), sink_max_head=sink.mean(0).max().item(),
                         consistency=consistency.mean().item(),
                         entropy=torch.cat(rec["entropy"]).mean().item(),
                         cls_mass=torch.cat(rec["cls"]).mean().item() if rec["cls"] else float("nan")))
    summary = dict(setting=label, dataset=dataset, operator=operator, layers=len(rows),
                   acc_1=100 * correct / total,
                   sink_mean=np.mean([r["sink_mean"] for r in rows]),
                   sink_max_layer=max(r["sink_mean"] for r in rows),
                   sink_max_head=max(r["sink_max_head"] for r in rows),
                   consistency_max=max(r["consistency"] for r in rows),
                   entropy=np.mean([r["entropy"] for r in rows]),
                   cls_mass=np.nanmean([r["cls_mass"] for r in rows]) if any(not math.isnan(r["cls_mass"]) for r in rows) else float("nan"),
                   rollout_sink=torch.cat(rollout_sink).mean().item(),
                   high_norm_frac=torch.cat(norm_frac).mean().item(),
                   norm_max_over_median=torch.cat(norm_ratio).mean().item())
    del model
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", nargs="*", help="subset of labels like 'Discrete/cifar100/MHA'")
    args = parser.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)

    layer_rows, summaries = [], []
    for label, dataset, operator, config, model_key, checkpoint in MODELS:
        tag = f"{label}/{dataset}/{operator}"
        if args.only and tag not in args.only:
            continue
        cfg = load_config(config, model_key, checkpoint)
        try:
            E.resolve_checkpoint(cfg)
        except (FileNotFoundError, StopIteration, KeyError) as error:
            print(f"skip {tag}: no checkpoint ({error})", flush=True)
            continue
        loader = (imagenet_loader(args.num_images, args.batch_size) if dataset == "imagenet"
                  else cifar_loader(cfg, args.num_images, args.batch_size))
        rows, summary = analyse(label, dataset, operator, cfg, loader, args.device)
        layer_rows += rows
        summaries.append(summary)
        print(f"{tag:32s} acc {summary['acc_1']:5.1f} sink mean {summary['sink_mean']:6.2f} max-layer {summary['sink_max_layer']:6.2f} "
              f"max-head {summary['sink_max_head']:7.2f} consist {summary['consistency_max']:.2f} "
              f"ent {summary['entropy']:.3f} cls {summary['cls_mass']:.4f} rollout {summary['rollout_sink']:6.2f} "
              f"highnorm {summary['high_norm_frac']:.4f} max/med {summary['norm_max_over_median']:.2f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_layer.csv", layer_rows), ("summary.csv", summaries)):
        if rows:
            with open(OUTPUT_DIR / name, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Wrote {OUTPUT_DIR}/per_layer.csv and summary.csv")


if __name__ == "__main__":
    main()
