#!/usr/bin/env python3
"""Radar plot: CIFAR-100 top-1 accuracy at D=768, dense (MHA) vs SSOG, one
axis per architecture setting (continuous, discrete, looped).

Data is hardcoded from the paper's own reported numbers (main.tex) plus two
numbers supplied directly by the user for runs finished on another cluster:
  - Continuous dense (ODEViT) @768: 70.47.
  - Looped dense (bViT) @768: 71.83.
  - Looped SSOG @768: 79.80.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

FAMILIES = ["Continuous", "Discrete", "Looped"]

DENSE = [70.47, 75.09, 71.83]
SSOG = [74.10, 79.02, 79.80]

DENSE_COLOR = "#E69F00"  # Okabe-Ito orange
SSOG_COLOR = "#0072B2"   # Okabe-Ito blue


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
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def main():
    configure_style()
    n = len(FAMILIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    dense = DENSE + DENSE[:1]
    ssog = SSOG + SSOG[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FAMILIES)

    ax.set_ylim(60, 82)
    ax.set_yticks([65, 70, 75, 80])
    ax.set_rlabel_position(30)

    ax.plot(angles, dense, color=DENSE_COLOR, linewidth=2.0, marker="o", label="Multi-head")
    ax.fill(angles, dense, color=DENSE_COLOR, alpha=0.12)

    ax.plot(angles, ssog, color=SSOG_COLOR, linewidth=2.0, marker="^", label="Proposed")
    ax.fill(angles, ssog, color=SSOG_COLOR, alpha=0.12)

    ax.set_title("CIFAR-100 top-1 accuracy", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9)

    fig.tight_layout()
    output_dir = "experiments/SSOG/figures/efficiency"
    import os

    os.makedirs(output_dir, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(f"{output_dir}/accuracy_radar_families.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote accuracy_radar_families.pdf / .png")


if __name__ == "__main__":
    main()
