"""Create per-NFE SSOG probability and Gaussian-atom GIFs."""

from __future__ import annotations

import argparse
import io
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR100

from models.SSOGViT import SSOG_ViT
from assets.visualizations.ssog_gif_render import render_atom_frames as pillow_atom_frames, render_probability_frames as pillow_probability_frames


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_config(name: str):
    with initialize(version_base=None, config_path="../../configs/classification"):  # relative to this file
        return compose(config_name=name)


def load_model(cfg, checkpoint_path: str, device: torch.device):
    model = SSOG_ViT(**cfg.modeling.inputs)
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False
    )
    state = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    fixed = {}
    for key, value in state.items():
        key = key.removeprefix("model.")
        if key not in model_state:
            continue
        # Tolerate a since-changed LayerNorm weight/bias shape ([1, D] in
        # checkpoints saved before the current models/SSOGViT.py vs. [D] now):
        # reshaping preserves the stored values and only changes broadcasting.
        target_shape = model_state[key].shape
        if value.shape != target_shape and value.numel() == model_state[key].numel():
            value = value.reshape(target_shape)
        fixed[key] = value
    model.load_state_dict(fixed, strict=True)
    return model.to(device).eval()


def get_samples(cfg, indices):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (cfg.modeling.inputs.img_size, cfg.modeling.inputs.img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = CIFAR100(
        root=str(Path(cfg.data.dataset.dataset_path).expanduser()),
        train=False,
        download=False,
    )
    samples = []
    for index in indices:
        image, label = dataset[index]
        samples.append((index, transform(image), label, dataset.classes[label]))
    return samples, dataset.classes


def denormalize(image):
    mean = image.new_tensor(MEAN)[:, None, None]
    std = image.new_tensor(STD)[:, None, None]
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def figure_frame(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def save_gif(frames, path: Path, duration: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=max(1, round(duration * 1000)), loop=0, optimize=False)


@torch.no_grad()
def trajectory(model, image):
    attention_inputs = []

    def capture(_module, inputs):
        attention_inputs.append(inputs[0].detach())

    hook = model.equation.attn.register_forward_pre_hook(capture)
    try:
        output = model(
            pixel_values=image[None],
            output_attentions=False,
            output_attentions_trajectory=False,
            regularize=False,
        )
    finally:
        hook.remove()

    hidden = output.hidden_states
    pooled = hidden[:, :, 0] if model.has_cls_token else hidden.mean(dim=2)
    probabilities = model.classifier(pooled).softmax(dim=-1)[:, 0]
    if len(attention_inputs) != hidden.shape[0] - 1:
        raise RuntimeError(
            "This visualizer currently requires a one-evaluation-per-transition "
            f"solver (received {len(attention_inputs)} evaluations for "
            f"{hidden.shape[0] - 1} transitions). Use the Euler configuration."
        )
    return probabilities.cpu(), attention_inputs


def effective_geometry(attention, x, query_y, query_x):
    """Return exact query fields, lambda, mu and sigma for every head/atom."""
    if attention.has_cls_token:
        x = x[:, 1:]
    b, gh, gw = x.shape[0], attention.grid_h, attention.grid_w
    h, r = attention.num_heads, attention.num_atoms
    grid = x.reshape(b, gh, gw, attention.dim)
    base_sigma = F.softplus(attention.raw_sigma) + 1.0e-4 + attention.sigma_floor
    positive = lambda value: F.softplus(value) + 1.0e-4

    if attention.lookat:
        query = grid[:, query_y, query_x]
        row = grid[:, query_y]
        if attention.steer_mu:
            row_mu = attention.mu_delta(row).reshape(b, gw, h, r, 2)
            query_mu = attention.mu_delta(query).reshape(b, h, r, 2)
            scale = positive(attention.raw_mu_delta_scale) * attention.max_offset
            mu_y_row = attention.mu[None, None, :, :, 0] + scale * torch.tanh(row_mu[..., 0])
            mu_x = attention.mu[None, :, :, 1] + scale * torch.tanh(query_mu[..., 1])
        else:
            mu_y_row = attention.mu[None, None, :, :, 0].expand(b, gw, -1, -1)
            mu_x = attention.mu[None, :, :, 1].expand(b, -1, -1)
        if attention.steer_sigma:
            row_sigma = attention.sigma_delta(row).reshape(b, gw, h, r, 2)
            query_sigma = attention.sigma_delta(query).reshape(b, h, r, 2)
            scale = positive(attention.raw_sigma_delta_scale)
            sigma_y_row = base_sigma[None, None, :, :, 0] * torch.exp(scale * torch.tanh(row_sigma[..., 0]))
            sigma_x = base_sigma[None, :, :, 1] * torch.exp(scale * torch.tanh(query_sigma[..., 1]))
        else:
            sigma_y_row = base_sigma[None, None, :, :, 0].expand(b, gw, -1, -1)
            sigma_x = base_sigma[None, :, :, 1].expand(b, -1, -1)
        if attention.steer_lambda:
            gate = attention.lambda_gate(query).reshape(b, h, r)
            lam = torch.softmax(
                attention.log_lambda[None]
                + positive(attention.raw_lambda_gate_scale) * torch.tanh(gate),
                dim=-1,
            )
        else:
            lam = torch.softmax(attention.log_lambda, dim=-1)[None].expand(b, -1, -1)
    else:
        mu_y_row = attention.mu[None, None, :, :, 0].expand(b, gw, -1, -1)
        mu_x = attention.mu[None, :, :, 1].expand(b, -1, -1)
        sigma_y_row = base_sigma[None, None, :, :, 0].expand(b, gw, -1, -1)
        sigma_x = base_sigma[None, :, :, 1].expand(b, -1, -1)
        lam = torch.softmax(attention.log_lambda, dim=-1)[None].expand(b, -1, -1)

    source_y = torch.arange(gh, device=x.device, dtype=x.dtype)
    source_x = torch.arange(gw, device=x.device, dtype=x.dtype)
    dy, dx = query_y - source_y, query_x - source_x
    temperature = F.softplus(attention.raw_temperature) + 0.5
    log_y = attention._log_kernel(
        dy[None, None, None, None, :], mu_y_row[..., None], sigma_y_row[..., None]
    )
    log_x = attention._log_kernel(
        dx[None, None, None, :], mu_x[..., None], sigma_x[..., None]
    )
    axis_y = torch.softmax(log_y / temperature, dim=-1)
    axis_x = torch.softmax(log_x / temperature, dim=-1)
    fields = axis_y.permute(0, 2, 3, 4, 1) * axis_x[..., None, :]

    # y-geometry varies with source x in steered SSOG; report its lambda-weighted
    # field center/width summaries without pretending it is a single Gaussian.
    mu_y = mu_y_row.mean(dim=1)
    sigma_y = sigma_y_row.mean(dim=1)
    mu = torch.stack((mu_y, mu_x), dim=-1)
    sigma = torch.stack((sigma_y, sigma_x), dim=-1)
    return fields[0], lam[0], mu[0], sigma[0]


def render_probability_frames(rgb, probabilities, class_names, true_label, top_k):
    final_top = probabilities[-1].topk(min(top_k, probabilities.shape[-1])).indices
    if true_label not in final_top:
        final_top = torch.cat((final_top[:-1], torch.tensor([true_label])))
    frames = []
    colors = ["#ef476f" if int(index) == true_label else "#118ab2" for index in final_top]
    for step in range(probabilities.shape[0]):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
        axes[0].imshow(rgb)
        axes[0].set_title(f"True class: {class_names[true_label]}")
        axes[0].axis("off")
        values = probabilities[step, final_top].numpy()
        labels = [class_names[int(index)] for index in final_top]
        axes[1].barh(np.arange(len(labels)), values, color=colors)
        axes[1].set_yticks(np.arange(len(labels)), labels)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, max(1.0, float(probabilities[:, final_top].max()) * 1.1))
        axes[1].set_xlabel("Class probability")
        axes[1].set_title(f"ODE classification trajectory — NFE {step}/{probabilities.shape[0]-1}")
        for y, value in enumerate(values):
            axes[1].text(value + 0.01, y, f"{value:.3f}", va="center")
        figure.tight_layout()
        frames.append(figure_frame(figure))
    return frames


@torch.no_grad()
def atom_frames(rgb, probabilities, inputs, attention, class_names, query):
    qy, qx = query
    heads, atoms = attention.num_heads, attention.num_atoms
    frames = []
    for step, x in enumerate(inputs, start=1):
        fields, weights, mu, sigma = effective_geometry(attention, x, qy, qx)
        figure = plt.figure(figsize=(3.0 * (atoms + 1), 2.35 * heads))
        grid = figure.add_gridspec(heads, atoms + 1, width_ratios=[1.15] + [1] * atoms)
        for head in range(heads):
            image_axis = figure.add_subplot(grid[head, 0])
            image_axis.imshow(rgb)
            mixture = (fields[head] * weights[head, :, None, None]).sum(dim=0)
            mixture = F.interpolate(mixture[None, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
            image_axis.imshow(mixture.detach().cpu().numpy(), cmap="magma", alpha=0.58)
            image_axis.set_title(f"head {head} mixture")
            image_axis.axis("off")
            for atom in range(atoms):
                axis = figure.add_subplot(grid[head, atom + 1])
                field = F.interpolate(fields[head, atom][None, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
                axis.imshow(rgb)
                axis.imshow(field.detach().cpu().numpy(), cmap="magma", alpha=0.62)
                axis.set_title(
                    f"H{head} A{atom}  λ={weights[head, atom]:.2f}\n"
                    f"μ=({mu[head, atom, 0]:.2f},{mu[head, atom, 1]:.2f}) "
                    f"σ=({sigma[head, atom, 0]:.2f},{sigma[head, atom, 1]:.2f})",
                    fontsize=8,
                )
                axis.axis("off")
        prediction = int(probabilities[step].argmax())
        mode = "steered" if attention.lookat else "fixed"
        figure.suptitle(
            f"SSOG {mode} atoms — NFE {step}/{len(inputs)}, query={query}, "
            f"prediction={class_names[prediction]} ({probabilities[step, prediction]:.3f})",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        frames.append(figure_frame(figure))
    return frames


def main(args):
    cfg = load_config(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    samples, class_names = get_samples(cfg, args.indices)
    output_dir = Path(args.output_dir)
    query = tuple(args.query) if args.query else (model.grid_h // 2, model.grid_w // 2)
    if not (0 <= query[0] < model.grid_h and 0 <= query[1] < model.grid_w):
        raise ValueError(f"query {query} lies outside {model.grid_h}x{model.grid_w} patch grid")

    metadata = {
        "model_config": args.model_config,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "indices": list(args.indices),
        "query": list(query),
        "lookat": model.equation.attn.lookat,
        "num_eval_steps": model.num_eval_steps,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(metadata), output_dir / "metadata.yaml")

    for index, image, label, class_name in samples:
        image = image.to(device)
        probabilities, inputs = trajectory(model, image)
        rgb = denormalize(image)
        prefix = output_dir / f"sample_{index:05d}_{class_name.replace(' ', '_')}"
        save_gif(
            pillow_probability_frames(
                rgb, probabilities, class_names, label, args.top_k
            ),
            Path(f"{prefix}_probabilities.gif"),
            args.duration,
        )
        save_gif(
            pillow_atom_frames(rgb, probabilities, inputs, model.equation.attn, class_names, query, effective_geometry),
            Path(f"{prefix}_atoms.gif"),
            args.duration,
        )
        print(f"Saved GIF pair for sample {index} ({class_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--query", type=int, nargs=2, default=None, metavar=("ROW", "COL"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--duration", type=float, default=0.7)
    parser.add_argument("--output-dir", default="artifacts/ssog_nfe_gifs")
    main(parser.parse_args())
