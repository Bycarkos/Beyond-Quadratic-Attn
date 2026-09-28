#!/usr/bin/env python3
"""Ablation sensitivity across the three architecture settings, wide layout.

One panel per ablated knob (Heads / Atoms / Depth), three lines per panel --
one per architecture. Accuracies are centred on each architecture's own block
mean, so all three settings share a single axis despite sitting 5-10 points
apart in absolute accuracy: the subject here is how far each knob *moves*
accuracy, not which setting scores higher (that comparison lives in the width
and dense-vs-SSoG figures).

Because the axis is shared, the vertical spread of a line is directly
readable as its sensitivity -- no per-panel annotation needed.

The shaded band is the 3-seed standard deviation measured on the Continuous
flagship configuration; anything inside it is within single-seed noise.

Numbers are final-epoch validation Acc@1 from Tables ssog-capacity,
ssog-capacity-discrete and ssog-capacity-looped. The Continuous Depth block's
NFE=16 point is omitted so all three settings span the same 6/8/12 grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("experiments/SSOG/figures/efficiency")

COLORS = {"Continuous": "#D55E00", "Discrete": "#0072B2", "Looped": "#009E73"}
MARKERS = {"Continuous": "o", "Discrete": "s", "Looped": "^"}

# Measured 3-seed spread on the Continuous flagship (74.22 / 70.68 / 71.06).
SEED_STD = 1.9

PANELS = [
    ("Heads", "Heads", [6, 12, 24], {
        "Continuous": [73.62, 74.10, 74.40],
        "Discrete": [79.97, 78.63, 79.52],
        "Looped": [79.32, 79.80, 79.51],
    }),
    ("Atoms", "Atoms per head ($R$)", [1, 2, 4, 8], {
        "Continuous": [73.42, 74.13, 74.10, 73.99],
        "Discrete": [79.62, 80.34, 78.63, 79.93],
        "Looped": [80.31, 79.67, 79.80, 79.73],
    }),
    ("Depth", "Depth / NFE", [6, 8, 12], {
        "Continuous": [69.70, 72.16, 73.99],
        "Discrete": [79.55, 79.70, 78.63],
        "Looped": [79.23, 78.88, 79.80],
    }),
]

ORDER = ["Continuous", "Discrete", "Looped"]

INK = "black"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"
BAND = "#b9b9b9"


def configure_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "axes.titleweight": "normal", "axes.titlepad": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.titlecolor": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# Direct in-plot labels instead of a legend box: (panel index, x index,
# vertical offset in accuracy points, vertical alignment).
DIRECT_LABELS = {
    "Continuous": (0, 1, -0.55, "top"),
    "Discrete": (0, 1, -0.55, "top"),
    "Looped": (0, 1, 0.5, "bottom"),
}


def main() -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1), sharey=True,
                             gridspec_kw={"wspace": 0.09})

    for i, (ax, (_, title, xs, series)) in enumerate(zip(axes, PANELS)):
        positions = range(len(xs))

        ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

        for arch in ORDER:
            ax.plot(positions, series[arch], color=COLORS[arch], marker=MARKERS[arch],
                    markersize=4.5, markerfacecolor=COLORS[arch],
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)

            panel, x_idx, dy, valign = DIRECT_LABELS[arch]
            if i == panel:
                ax.text(x_idx, series[arch][x_idx] + dy, arch, color=COLORS[arch],
                        fontsize=8, ha="center", va=valign, zorder=4)

        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlim(-0.28, len(xs) - 0.72)
        ax.set_title(title, pad=5)
        ax.tick_params(length=2.5, pad=2)

    axes[0].set_ylim(68.8, 81.4)
    axes[0].set_ylabel("Acc@1 (%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"ablation_horizontal.{suffix}")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'ablation_horizontal.pdf'} / .png")


if __name__ == "__main__":
    main()
