#!/usr/bin/env python3
"""Width scaling on CIFAR-100 (tab:ssog-vs-dense-width) as a figure: Acc@1 vs
parameter count, one panel per architecture setting. Each line joins one
operator's three widths (D = 384, 576, 768), so up-and-left means more
accurate with fewer parameters. The dotted connector marks the D = 768 gap.

Styled like the ablation prototype (plot_ablation_horizontal.py)."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from assets.plotting.plot_ablation_horizontal import GRID, MUTED, configure_style

OUTPUT = Path("experiments/SSOG/figures/efficiency")
WIDTHS = (384, 576, 768)
MHA_COLOR, SSOG_COLOR = "#D55E00", "#0072B2"  # Okabe-Ito, as in efficiency_delta_by_resolution
INK = "black"

# setting: (MHA params M, SSoG params M, MHA Acc@1, SSoG Acc@1), per width
DATA = {
    "Continuous": ((3.07, 6.59, 11.45), (2.8, 5.9, 10.3), (69.15, 69.36, 70.47), (71.37, 73.48, 74.10)),
    "Discrete": ((21.69, 48.47, 85.86), (18.5, 41.0, 72.3), (74.51, 74.20, 74.67), (77.56, 78.18, 78.63)),
    "Looped": ((2.18, 4.60, 7.91), (1.9, 3.9, 6.7), (72.04, 71.24, 72.30), (77.46, 79.38, 79.80)),
}


def series(ax, params, accs, color, marker, linestyle):
    ax.plot(params, accs, color=color, linestyle=linestyle, marker=marker,
            markersize=4.5, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3)


def gap(ax, mha_p, mha_a, ssog_p, ssog_a):
    """Dotted connector between the D=768 points, labelled with the Acc@1 gain."""
    ax.plot([mha_p, ssog_p], [mha_a, ssog_a], color=MUTED, linestyle=(0, (1, 1.6)),
            linewidth=0.8, zorder=2)
    ax.text((mha_p * ssog_p) ** 0.5 / 1.1, (mha_a + ssog_a) / 2, f"+{ssog_a - mha_a:.1f}",
            color=INK, fontsize=7.5, ha="right", va="center")


def main():
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1), sharey=True)
    for i, (ax, (setting, (mha_p, ssog_p, mha_a, ssog_a))) in enumerate(zip(axes, DATA.items())):
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        gap(ax, mha_p[-1], mha_a[-1], ssog_p[-1], ssog_a[-1])
        series(ax, mha_p, mha_a, MHA_COLOR, "o", "--")
        series(ax, ssog_p, ssog_a, SSOG_COLOR, "^", "-")

        ax.set_xscale("log")
        ax.set_xlim(min(ssog_p) / 1.5, max(mha_p) * 1.3)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}M"))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs="all"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_title(setting, pad=5)
        ax.tick_params(length=2.5, pad=2)

        if i == 0:
            mid_p = (ssog_p[0] * ssog_p[1]) ** 0.5  # geometric midpoint on the log axis
            ax.text(mid_p, (ssog_a[0] + ssog_a[1]) / 2 - 0.7, "SSoG", color=SSOG_COLOR,
                    fontsize=8, ha="left", va="top")
            ax.text(mha_p[0] * 0.93, mha_a[0] - 0.55, "MHA", color=MHA_COLOR,
                    fontsize=8, ha="left", va="top")
            for p, a, d in zip(ssog_p, ssog_a, WIDTHS):
                label = f"$D{{=}}{d}$" if d == WIDTHS[0] else f"{d}"
                ax.text(p, a + 0.6, label, color=INK, fontsize=6.5, ha="center", va="bottom")

    axes[0].set_ylabel("Acc@1 (%)")
    axes[0].set_ylim(67.5, 81.5)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[1].set_xlabel("Parameters", labelpad=3)
    fig.tight_layout(w_pad=1.2)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"width_scaling.{suffix}")
    plt.close(fig)
    print("Wrote width_scaling.pdf / .png")


if __name__ == "__main__":
    main()
