#!/usr/bin/env python3
"""Qualitative SSoG attention at different patch sizes (paper figure).

For the best CIFAR-100 model (looped SSoG) and the best ImageNet model
(discrete SSoG, one layer), each row shows one head's attention from a fixed
query pixel over a validation image; columns are patch 16 (training), 14, and
8 with the learned geometry kept in token units, and 8 with mu/sigma rescaled
to the new grid. Models are rebuilt with eval_patch_size.build_model, so the
plotted kernels are exactly the ones evaluated in tab:patch-size-robustness.
Fixed SSoG kernels do not depend on the input, so no forward pass is needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from torchvision import transforms
from torchvision.datasets import CIFAR100

import eval_patch_size as E
from models.SSOGViT import SSOGAttention
from tests.ssog_experiments.common import find_imagenet_arrow_shards

matplotlib.use("Agg")

OUTPUT = Path("experiments/SSOG/figures/attention")
QUERY_PIXEL = (116, 116)  # (y, x); inside the centre token at every patch size
# Figure variants: file suffix -> query pixels. The multi-query set mirrors the
# trainer's visualization_queries ([3,3], [3,10], [7,7], [10,3], [10,10] on the
# 14x14 grid); each pixel sits inside one token at every patch size.
QUERY_SETS = {
    "": (QUERY_PIXEL,),
    "_multiquery": ((58, 58), (58, 170), QUERY_PIXEL, (170, 58), (170, 170)),
}
COLUMNS = ((16, "token", "Patch 16\n(training)"), (14, "token", "Patch 14"),
           (8, "token", "Patch 8"), (8, "image", "Patch 8\n(rescaled $\\mu,\\sigma$)"))
ATTENTION_COLOR = "#D55E00"  # single-hue sequential overlay (Okabe-Ito vermillion)

MODELS = {
    "cifar100_looped": dict(
        config="evaluation/patch_size/cifar100/looped.yaml", layer=0,
        title="CIFAR-100, looped SSoG", images=(3, 17, 42, 88),
    ),
    "imagenet_discrete": dict(
        config="evaluation/patch_size/imagenet/discrete.yaml", layer=6,
        title="ImageNet, discrete SSoG (layer 7 of 12)", images=(11, 23, 57, 131),
    ),
}


def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times"],
        "font.size": 9, "axes.titlesize": 9, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def load_images(cfg, indices):
    resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    if cfg.data.dataset.name == "cifar100":
        dataset = CIFAR100(root=cfg.data.dataset.dataset_path, train=False, download=False)
        pil = [dataset[i][0] for i in indices]
    else:
        from datasets import Dataset
        dataset = Dataset.from_file(str(find_imagenet_arrow_shards()[0]))
        pil = [dataset[i]["image"] for i in indices]
    return [np.asarray(resize(image.convert("RGB")), dtype=np.float32) / 255 for image in pil]


def head_maps(attn, query_pixel, patch_size):
    """Per-head [H, G, G] attention from the query token to every source token."""
    with torch.no_grad():
        ay, ax, lam = attn._compute_fixed_kernel(*attn._displacements(torch.empty(0)))
        qi, qj = (p // patch_size for p in query_pixel)
        return torch.einsum("hr,hri,hrj->hij", lam, ay[:, :, qi], ax[:, :, qj]).numpy()


def pick_heads(maps, count=4):
    """Diverse heads from the training-patch maps: broadest, most shifted,
    most vertical, most horizontal (distinct)."""
    g = maps.shape[-1]
    ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    my = (maps * ys).sum((1, 2)); mx = (maps * xs).sum((1, 2))
    vy = (maps * (ys - my[:, None, None]) ** 2).sum((1, 2))
    vx = (maps * (xs - mx[:, None, None]) ** 2).sum((1, 2))
    centre = g // 2
    shift = np.hypot(my - centre, mx - centre)
    order = [np.argmax(vy + vx), np.argmax(shift), np.argmax(vy / vx), np.argmax(vx / vy)]
    heads = []
    for h in order + list(range(len(maps))):
        if h not in heads:
            heads.append(int(h))
    return heads[:count]


def overlay(ax, image, attentions, queries, patch_size):
    """attentions: one [G, G] map per query; each is normalized to its own
    maximum and the panel shows their pointwise maximum."""
    gray = image.mean(-1, keepdims=True) * 0.85 + 0.1
    combined = np.max([a / a.max() for a in attentions], axis=0)
    weight = np.kron(combined, np.ones((patch_size, patch_size)))[..., None]
    ax.imshow(gray * (1 - weight) + np.array(to_rgb(ATTENTION_COLOR)) * weight)
    for query in queries:
        qy, qx = ((p // patch_size) * patch_size for p in query)
        ax.add_patch(Rectangle((qx - 0.5, qy - 0.5), patch_size, patch_size,
                               fill=False, edgecolor="white", linewidth=1.0))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_model(key, spec):
    with initialize(version_base=None, config_path="../../configs/classification"):  # relative to this file
        cfg = compose(config_name=spec["config"])
    state = torch.load(E.resolve_checkpoint(cfg), map_location="cpu", weights_only=False)["state_dict"]
    images = load_images(cfg, spec["images"])

    attns = {}
    for patch, geometry, _ in COLUMNS:
        model = E.build_model(cfg, state, patch, geometry)
        attns[patch, geometry] = [m for m in model.modules() if isinstance(m, SSOGAttention)][spec["layer"]]
    heads = pick_heads(head_maps(attns[16, "token"], QUERY_PIXEL, 16), count=len(images))
    for suffix, queries in QUERY_SETS.items():
        maps = {
            (patch, geometry): [head_maps(attns[patch, geometry], q, patch) for q in queries]
            for patch, geometry, _ in COLUMNS
        }
        draw(f"{key}{suffix}", spec["title"], heads, images, maps, queries)


def draw(name, title, heads, images, maps, queries):
    fig, axes = plt.subplots(len(heads), len(COLUMNS), figsize=(5.5, 5.6))
    for row, (head, image) in enumerate(zip(heads, images)):
        for col, (patch, geometry, label) in enumerate(COLUMNS):
            ax = axes[row, col]
            overlay(ax, image, [m[head] for m in maps[patch, geometry]], queries, patch)
            if row == 0:
                ax.set_title(label)
        axes[row, 0].set_ylabel(f"Head {head + 1}")
    fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=(0, 0, 1, 0.96))
    fig.suptitle(title, y=0.985)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"patch_size_attention_{name}.{suffix}")
    plt.close(fig)
    print(f"Wrote patch_size_attention_{name}.pdf/.png (heads {[h + 1 for h in heads]})")


def main():
    configure_style()
    for key, spec in MODELS.items():
        plot_model(key, spec)


if __name__ == "__main__":
    main()
