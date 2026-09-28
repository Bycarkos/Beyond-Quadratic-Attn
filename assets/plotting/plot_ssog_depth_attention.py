#!/usr/bin/env python3
"""Qualitative SSoG attention along depth (paper figure), on colour images.

Discrete SSoG (best ImageNet model): every block has its own heads, so each
column is one block and each row one head *role* (broadest, most shifted,
most vertical, most horizontal) chosen per block; the tag in each panel is the
head that fills that role in that block.

Looped SSoG (best CIFAR-100 model): one block is reused at every step and the
fixed kernel depends only on learned parameters, so the single-step kernel is
identical at all steps. What changes along the steps is the cumulative mixing,
shown as attention rollout with the residual path (Abnar & Zuidema, 2020):
R_k = prod_{s<=k} (A_h + I) / 2, per head (rows are the same heads at every
step). It ignores the MLP and cross-head mixing.

Attention from the centre query token (white box) is shown by dimming the
image outside the attended region, with white contours at 20% and 50% of the
panel maximum. The query token's own weight is capped at the largest weight on
any other token, so the residual self-term does not hide the spread.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from matplotlib.patches import Rectangle

import eval_patch_size as E
from models.SSOGViT import SSOGAttention
from assets.plotting.plot_ssog_patch_size_attention import QUERY_PIXEL, configure_style, load_images

matplotlib.use("Agg")

OUTPUT = Path("experiments/SSOG/figures/attention")
PATCH = 16
ROLES = ("Broadest", "Most shifted", "Most vertical", "Most horizontal")
COLUMNS_PER_HALF = 6

MODELS = {
    "imagenet_discrete": dict(
        config="evaluation/patch_size/imagenet/discrete.yaml", mode="blocks",
        title="ImageNet, discrete SSoG: kernel of each block", image=57, label="Block",
    ),
    "cifar100_looped": dict(
        config="evaluation/patch_size/cifar100/looped.yaml", mode="steps",
        title="CIFAR-100, looped SSoG: cumulative mixing after each step", image=3, label="Step",
    ),
}


def dense_kernels(attn):
    """Per-head [H, N, N] patch-to-patch attention matrices of a fixed SSoG block."""
    with torch.no_grad():
        ay, ax, lam = attn._compute_fixed_kernel(*attn._displacements(torch.empty(0)))
        a = torch.einsum("hr,hrai,hrbj->habij", lam, ay, ax)
    h, g = a.shape[0], a.shape[1]
    return a.reshape(h, g * g, g * g).double().numpy(), g


def role_ranking(maps):
    """For each role, heads ranked best-first. maps: [H, G, G] from one query."""
    g = maps.shape[-1]
    ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    my = (maps * ys).sum((1, 2)); mx = (maps * xs).sum((1, 2))
    vy = (maps * (ys - my[:, None, None]) ** 2).sum((1, 2))
    vx = (maps * (xs - mx[:, None, None]) ** 2).sum((1, 2))
    shift = np.hypot(my - g // 2, mx - g // 2)
    scores = (vy + vx, shift, vy / vx, vx / vy)
    return [list(np.argsort(-s)) for s in scores]


def assign_roles(maps):
    """One distinct head per role; a role whose best head is taken gets its next best."""
    heads = []
    for ranking in role_ranking(maps):
        heads.append(int(next(h for h in ranking if h not in heads)))
    return heads


def spotlight(ax, image, attention, tag):
    g = attention.shape[-1]
    qi, qj = (p // PATCH for p in QUERY_PIXEL)
    attention = attention.copy()
    others = np.delete(attention.ravel(), qi * g + qj)
    attention[qi, qj] = min(attention[qi, qj], others.max())
    attention = attention / attention.max()
    weight = np.kron(attention, np.ones((PATCH, PATCH)))[..., None]
    ax.imshow(image * (0.35 + 0.65 * weight))
    smooth = torch.nn.functional.interpolate(
        torch.from_numpy(attention)[None, None], size=image.shape[:2],
        mode="bilinear", align_corners=False)[0, 0].numpy()
    ax.contour(smooth, levels=(0.2, 0.5), colors="white", linewidths=(0.5, 0.9))
    qy, qx = ((p // PATCH) * PATCH for p in QUERY_PIXEL)
    ax.add_patch(Rectangle((qx - 0.5, qy - 0.5), PATCH, PATCH,
                           fill=False, edgecolor="white", linewidth=0.8))
    ax.text(0.04, 0.96, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color="white")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_panels(cfg, state, mode):
    """Returns [(column_index, [(head, [G, G] map) per role])]."""
    model = E.build_model(cfg, state, PATCH, "token")
    attns = [m for m in model.modules() if isinstance(m, SSOGAttention)]
    q = (QUERY_PIXEL[0] // PATCH, QUERY_PIXEL[1] // PATCH)
    columns = []
    if mode == "blocks":
        for attn in attns:
            kernels, g = dense_kernels(attn)
            maps = kernels[:, q[0] * g + q[1]].reshape(-1, g, g)
            columns.append([(h, maps[h]) for h in assign_roles(maps)])
    else:
        kernels, g = dense_kernels(attns[0])
        heads = assign_roles(kernels[:, q[0] * g + q[1]].reshape(-1, g, g))
        steps = cfg.modeling.inputs.depth
        residual = 0.5 * (kernels[heads] + np.eye(g * g))  # attention + skip connection
        rows = np.zeros((len(heads), g * g)); rows[:, q[0] * g + q[1]] = 1.0
        for _ in range(steps):
            rows = np.einsum("hn,hnm->hm", rows, residual)  # one more step
            columns.append([(h, rows[i].reshape(g, g)) for i, h in enumerate(heads)])
    return columns


def plot_model(key, spec):
    with initialize(version_base=None, config_path="../../configs/classification"):  # relative to this file
        cfg = compose(config_name=spec["config"])
    state = torch.load(E.resolve_checkpoint(cfg), map_location="cpu", weights_only=False)["state_dict"]
    image = load_images(cfg, (spec["image"],))[0]
    columns = build_panels(cfg, state, spec["mode"])

    halves = [columns[i:i + COLUMNS_PER_HALF] for i in range(0, len(columns), COLUMNS_PER_HALF)]
    fig = plt.figure(figsize=(5.5, 0.95 * len(ROLES) * len(halves) + 0.9), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.01, hspace=0.02, wspace=0.01)
    for half_index, (subfig, half) in enumerate(zip(fig.subfigures(len(halves), 1, hspace=0.04), halves)):
        axes = subfig.subplots(len(ROLES), COLUMNS_PER_HALF)
        for col, panels in enumerate(half):
            index = half_index * COLUMNS_PER_HALF + col
            for role_index, (head, attention) in enumerate(panels):
                spotlight(axes[role_index, col], image, attention, f"H{head + 1}")
            axes[0, col].set_title(f"{spec['label']} {index + 1}", fontsize=8, pad=3)
        for role_index, role in enumerate(ROLES):
            axes[role_index, 0].set_ylabel(role, fontsize=7.5)
    fig.suptitle(spec["title"], fontsize=9)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"depth_attention_{key}.{suffix}")
    plt.close(fig)
    print(f"Wrote depth_attention_{key}.pdf/.png")


def main():
    configure_style()
    for key, spec in MODELS.items():
        plot_model(key, spec)


if __name__ == "__main__":
    main()
