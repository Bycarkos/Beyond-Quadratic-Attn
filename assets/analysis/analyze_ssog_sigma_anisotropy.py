"""Axis-specific sigma-x/sigma-y and anisotropy-ratio statistics for SSOG checkpoints.

Addresses ablation_report.md section 14: "Add separate sigma-x, sigma-y, and
anisotropy-ratio statistics [...] for identical samples, queries, and
checkpoints." The existing aggregate val/ssog_sigma_min/mean/max metrics mix
both spatial axes together; this script splits them and adds an explicit
anisotropy ratio (max(sigma_y, sigma_x) / min(sigma_y, sigma_x)) per head/atom.

For fixed SSOG (lookat=False), sigma has no query dependence, so the base
sigma tensor read directly from the checkpoint IS the effective sigma
everywhere -- no forward pass over data is required. For content-steered
SSOG (lookat=True), sigma is query-dependent, so this script reuses
`effective_geometry` from visualize_ssog_nfe.py to average over a fixed set
of validation samples and query positions, exactly mirroring the qualitative
per-sample maps that script already produces for the same checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from assets.visualizations.visualize_ssog_nfe import effective_geometry, get_samples, load_config, load_model


def base_sigma_tensor(attn) -> torch.Tensor:
    """Sigma with no query dependence: shape (heads, atoms, 2) as (y, x)."""
    return F.softplus(attn.raw_sigma) + 1.0e-4 + attn.sigma_floor


def axis_stats(sigma_y: torch.Tensor, sigma_x: torch.Tensor) -> dict:
    ratio = torch.maximum(sigma_y, sigma_x) / torch.minimum(sigma_y, sigma_x)
    def summarize(t):
        return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}
    return {
        "sigma_y": summarize(sigma_y),
        "sigma_x": summarize(sigma_x),
        "anisotropy_ratio": summarize(ratio),
        "n_head_atom_entries": int(sigma_y.numel()),
    }


@torch.no_grad()
def analyze_fixed(attn) -> dict:
    sigma = base_sigma_tensor(attn)  # (H, R, 2)
    return axis_stats(sigma[..., 0], sigma[..., 1])


@torch.no_grad()
def analyze_checkpoint(label: str, model_config: str, checkpoint: str, samples, query, device) -> dict:
    cfg = load_config(model_config)
    model = load_model(cfg, checkpoint, device)
    attn = model.equation.attn

    if not attn.lookat:
        stats = analyze_fixed(attn)
        stats["mode"] = "fixed (query-independent, exact)"
        return {"label": label, **stats}

    qy, qx = query
    all_sy, all_sx = [], []

    def capture_grid(image):
        captured = {}

        def hook(_module, inputs):
            captured["x"] = inputs[0].detach()

        handle = model.equation.attn.register_forward_pre_hook(hook)
        try:
            model(pixel_values=image[None], output_attentions=False,
                  output_attentions_trajectory=False, regularize=False)
        finally:
            handle.remove()
        return captured["x"]

    for _, image, _, _ in samples:
        image = image.to(device)
        x = capture_grid(image)
        _, _, _, sigma = effective_geometry(attn, x, qy, qx)  # (H, R, 2)
        all_sy.append(sigma[..., 0])
        all_sx.append(sigma[..., 1])

    sigma_y = torch.stack(all_sy).mean(dim=0)
    sigma_x = torch.stack(all_sx).mean(dim=0)
    stats = axis_stats(sigma_y, sigma_x)
    stats["mode"] = f"steered (query-dependent, averaged over {len(samples)} samples @ query={query})"
    return {"label": label, **stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="label=model_config=checkpoint_path triples, ':' separated")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--query", type=int, nargs=2, default=[7, 7], metavar=("ROW", "COL"))
    parser.add_argument("--output", default="experiments/SSOG/sigma_anisotropy/summary.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    samples_cache = {}
    for spec in args.checkpoints:
        label, model_config, checkpoint = spec.split(":", 2)
        cfg = load_config(model_config)
        key = cfg.data.dataset.dataset_path
        if key not in samples_cache:
            samples_cache[key], _ = get_samples(cfg, args.indices)
        result = analyze_checkpoint(label, model_config, checkpoint, samples_cache[key], tuple(args.query), device)
        results.append(result)
        print(json.dumps(result, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "query": list(args.query), "indices": list(args.indices), "results": results,
    }, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
