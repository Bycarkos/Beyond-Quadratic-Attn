#!/usr/bin/env python3
"""Accuracy vs. throughput vs. parameter-count Pareto figure.

x = batch-8 throughput (images/s, NVIDIA A40, FP16), y = CIFAR-100 final
Top-1 accuracy, marker area proportional to parameter count. Each model
contributes one point per resolution (224 and 512 px); a thin line connects
a model's two resolution points to show how throughput moves as accuracy
stays fixed (accuracy is a property of the trained weights, not of the
inference resolution used for the efficiency benchmark).

Data provenance (all verified against ablation_report.md / main.tex):
  - Throughput and parameter counts: main.tex Table "Architecture scale and
    estimated computation" / "Batch-8 throughput and peak allocated memory"
    (section 11.4), a single hardware-matched NVIDIA A40 sweep. ODEViT is
    deliberately excluded: its dim768 accuracy (the width used in that sweep)
    has not finished training (job 13074, running as of 2026-09-14).
  - Accuracy: discrete dense ViT final Top-1 at dim192/384/768 (section 4.3,
    models/MatchedViT.py, seed 42) for the three ViT sizes -- these are the
    exact dim/head configuration used for the ViT-Tiny/Small/Base efficiency
    benchmark rows, so the pairing is exact, not approximate. Fixed SSOG-ODEViT
    (dim768/H12/R4/NFE12) uses the 3-seed mean +/- std (72.0 +/- 1.9, seeds
    42/43/44) rather than a single seed's number (section 5.1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("experiments/SSOG/figures/efficiency")

# Colors match the palette already used throughout the paper's other
# efficiency figures (Okabe-Ito colorblind-safe set): SSOG blue, ViT grey.
COLOR_SSOG = "#0072B2"
COLOR_VIT = "#4D4D4D"
GRID_GRAY = "#B5B5B5"

# (label, color, marker, accuracy%, accuracy_err, [(resolution, throughput_img_s, params_M), ...])
MODELS = [
    (
        "SSOG-ODEViT (ours)", COLOR_SSOG, "o", 72.0, 1.9,
        [(224, 289.79, 10.27), (512, 125.71, 10.90)],
    ),
    (
        "ViT-Tiny", COLOR_VIT, "^", 73.42, None,
        [(224, 983.08, 5.54), (512, 882.88, 5.70)],
    ),
    (
        "ViT-Small", COLOR_VIT, "^", 74.51, None,
        [(224, 1009.93, 21.70), (512, 374.24, 22.02)],
    ),
    (
        "ViT-Base", COLOR_VIT, "^", 74.67, None,
        [(224, 653.98, 85.88), (512, 126.91, 86.51)],
    ),
]


def configure_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRID_GRAY,
            "grid.color": GRID_GRAY,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
        }
    )


def area_for_params(params_m: float, min_pt: float = 60.0, max_pt: float = 1400.0) -> float:
    """Marker area (points^2) proportional to parameter count, not radius,
    scaled across the full 5.5-86M range so bubble AREA reads correctly."""
    lo, hi = 5.5, 86.0
    frac = (params_m - lo) / (hi - lo)
    return min_pt + frac * (max_pt - min_pt)


# Per-model label offset in points (dx, dy) from the 224px marker, tuned by
# hand against the rendered figure to keep the four labels clear of markers,
# the error bar, and each other.
LABEL_OFFSET = {
    "SSOG-ODEViT (ours)": (8, 9),
    "ViT-Tiny": (8, -13),
    "ViT-Small": (-58, 26),
    "ViT-Base": (-58, 11),
}


def main() -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    for label, color, marker, acc, acc_err, points in MODELS:
        xs = [p[1] for p in points]
        sizes = [area_for_params(p[2]) for p in points]
        ax.annotate(
            "", xy=(xs[1], acc), xytext=(xs[0], acc),
            arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.5, linewidth=1.3,
                             shrinkA=8, shrinkB=8),
            zorder=1,
        )
        if acc_err:
            ax.errorbar(
                [xs[0]], [acc], yerr=acc_err, color=color,
                linewidth=1.3, capsize=3, zorder=2, fmt="none",
            )
        # 224px: solid marker. 512px: faded marker (same color, lower alpha).
        ax.scatter(xs[0], acc, s=sizes[0], c=color, marker=marker,
                   edgecolors="white", linewidths=1.1, alpha=0.95, zorder=3)
        ax.scatter(xs[1], acc, s=sizes[1], c=color, marker=marker,
                   edgecolors="white", linewidths=1.1, alpha=0.4, zorder=3)
        dx, dy = LABEL_OFFSET[label]
        ax.annotate(
            label, (xs[0], acc), xytext=(dx, dy),
            textcoords="offset points", fontsize=7.8, color=color, fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Batch-8 throughput (images/s, NVIDIA A40, FP16) -- log scale")
    ax.set_ylabel("CIFAR-100 final Top-1 accuracy (%)")
    ax.set_ylim(68.5, 77.5)
    ax.set_xlim(100, 1400)
    ax.set_title("Accuracy vs. throughput vs. parameter count", fontsize=10)

    # Marker-size legend (bubble area = parameter count), placed in the one
    # region with no data (lower right: high throughput, low accuracy).
    size_ref_params = [5.5, 21.7, 85.9]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="black", markersize=np.sqrt(area_for_params(p)) * 0.9,
               label=f"{p:.1f} M params")
        for p in size_ref_params
    ]
    style_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="black",
               markeredgecolor="white", alpha=0.95, markersize=6, label="224 px (solid)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="black",
               markeredgecolor="white", alpha=0.4, markersize=6, label="512 px (faded)"),
    ]
    family_handles = [
        Line2D([0], [0], marker="^", linestyle="none", color=COLOR_VIT, markersize=6,
               label="Discrete dense ViT (matched depth/width)"),
        Line2D([0], [0], marker="o", linestyle="none", color=COLOR_SSOG, markersize=6,
               label="Fixed SSOG-ODEViT (3-seed mean $\\pm$ std)"),
    ]
    legend1 = ax.legend(handles=family_handles + style_handles, loc="upper right",
                         frameon=False, handletextpad=0.6, fontsize=7.3)
    ax.add_artist(legend1)
    ax.legend(handles=size_handles, loc="lower right", frameon=False,
              title="Marker area $\\propto$ params", title_fontsize=7.5,
              labelspacing=1.3, handletextpad=1.4, borderpad=0.8)

    fig.text(
        0.01, -0.02,
        "ODEViT (dense attention, same ODE architecture) omitted: its dim768 accuracy is still training (job 13074).\n"
        "Throughput from a single A40 run per configuration (section 11.4); repeat sweeps in progress (section 14).",
        fontsize=6.3, color="#666666", va="top",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"accuracy_throughput_pareto.{ext}")
    print(f"Wrote {OUTPUT_DIR / 'accuracy_throughput_pareto.pdf'}")


if __name__ == "__main__":
    main()
