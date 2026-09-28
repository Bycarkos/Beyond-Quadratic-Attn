#!/usr/bin/env python3
"""Capacity/depth ablation sensitivity across the three architecture settings.

3x3 grid: rows are the ablated knob (Heads / Atoms / Depth), columns are the
architecture (Continuous / Discrete / Looped). Accuracy is shared within a
column, free across columns -- the three settings sit ~5-10 points apart in
absolute accuracy, so one shared axis would flatten the ~1-point within-block
variation that is the actual subject here. The per-panel swing annotation
(max - min) is what makes sensitivity comparable across columns despite the
independent scales.

The Continuous Depth block's NFE=16 point is omitted so all three settings
span the same 6/8/12 grid.

Numbers are final-epoch validation Acc@1, from Tables ssog-capacity,
ssog-capacity-discrete and ssog-capacity-looped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("experiments/SSOG/figures/efficiency")

# architecture -> colour (same palette as the visual abstract)
COLORS = {"Continuous": "#E69F00", "Discrete": "#0072B2", "Looped": "#009E73"}

# ablation -> architecture -> (x values, Acc@1 values)
DATA = {
    "Heads": {
        "Continuous": ([6, 12, 24], [73.62, 74.10, 74.40]),
        "Discrete": ([6, 12, 24], [79.97, 78.63, 79.52]),
        "Looped": ([6, 12, 24], [79.32, 79.80, 79.51]),
    },
    "Atoms": {
        "Continuous": ([1, 2, 4, 8], [73.42, 74.13, 74.10, 73.99]),
        "Discrete": ([1, 2, 4, 8], [79.62, 80.34, 78.63, 79.93]),
        "Looped": ([1, 2, 4, 8], [80.31, 79.67, 79.80, 79.73]),
    },
    "Depth": {
        "Continuous": ([6, 8, 12], [69.70, 72.16, 73.99]),
        "Discrete": ([6, 8, 12], [79.55, 79.70, 78.63]),
        "Looped": ([6, 8, 12], [79.23, 78.88, 79.80]),
    },
}

ROWS = ["Heads", "Atoms", "Depth"]
COLUMNS = ["Continuous", "Discrete", "Looped"]

INK = "#333333"
MUTED = "#999999"


def configure_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-white")
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
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": False,
        }
    )


def main() -> None:
    configure_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4))

    # Shared y-limits per column (architecture), with a little headroom.
    ylims = {}
    for arch in COLUMNS:
        values = [v for row in ROWS for v in DATA[row][arch][1]]
        low, high = min(values), max(values)
        pad = max(0.35, (high - low) * 0.22)
        ylims[arch] = (low - pad, high + pad)

    for i, row in enumerate(ROWS):
        for j, arch in enumerate(COLUMNS):
            ax = axes[i, j]
            xs, ys = DATA[row][arch]
            positions = range(len(xs))

            ax.plot(positions, ys, color=COLORS[arch], marker="o",
                    markerfacecolor="white", markeredgewidth=1.4)

            ax.set_xticks(list(positions))
            ax.set_xticklabels([str(x) for x in xs])
            ax.set_xlim(-0.35, len(xs) - 0.65)
            ax.set_ylim(*ylims[arch])
            ax.tick_params(length=2.5)

            # Swing: what makes columns comparable despite independent scales.
            # Park it in whichever right-hand corner the curve leaves free --
            # a curve ending high frees the bottom, one ending low frees the top.
            ends_high = abs(ys[-1] - max(ys)) < abs(ys[-1] - min(ys))
            y_pos, v_align = (0.08, "bottom") if ends_high else (0.94, "top")
            ax.text(0.96, y_pos, f"$\\Delta$ {max(ys) - min(ys):.2f}",
                    transform=ax.transAxes, ha="right", va=v_align,
                    fontsize=8, color=MUTED)

            if i == 0:
                ax.set_title(arch, color=COLORS[arch], pad=8)
            if j == 0:
                ax.set_ylabel(row)

    fig.supylabel("Acc@1 (%)", fontsize=10, x=0.005)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"ablation_grid.{suffix}")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'ablation_grid.pdf'} / .png")


if __name__ == "__main__":
    main()
