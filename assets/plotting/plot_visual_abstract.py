#!/usr/bin/env python3
"""Visual abstract: accuracy radar (left) next to the architecture-scaling
efficiency scatter at resolution 512 (right), sharing one style/typography
pass so the two panels read as a single figure.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

from assets.plotting import plot_efficiency_results as eff
from assets.plotting import plot_accuracy_radar as radar
from assets.plotting.plot_ablation_horizontal import GRID, configure_style  # shared paper style

# Local-only renames: eff.FAMILY_LABELS/OPERATOR_LABELS also drive the other
# efficiency figures in plot_efficiency_results.py, so this visual abstract
# keeps its own copies instead of editing those shared dicts.
LOCAL_FAMILY_LABELS = {"odevit": "Continuous", "vit": "Discrete", "bvit": "Looped"}
LOCAL_OPERATOR_LABELS = {"dense": "Multi-head", "ssog": "Proposed"}
# Looped uses Okabe-Ito bluish green instead of the shared reddish purple,
# which read too close to the Continuous orange; all pairs keep OKLab
# Delta E >= 8.6 under protan/deutan/tritan simulation.
LOCAL_FAMILY_COLORS = {"odevit": "#E69F00", "vit": "#0072B2", "bvit": "#009E73"}
LOCAL_OPERATOR_ORDER = ("ssog", "dense")  # proposed operator listed first
# Operator colours as in the width-scaling and resolution-delta figures.
MHA_COLOR, SSOG_COLOR = "#D55E00", "#0072B2"
MARKER = dict(markersize=4.5, markeredgecolor="white", markeredgewidth=1.0)


def draw_radar(ax):
    n = len(radar.FAMILIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    dense = radar.DENSE + radar.DENSE[:1]
    ssog = radar.SSOG + radar.SSOG[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar.FAMILIES)
    ax.set_ylim(60, 82)
    ax.set_yticks([65, 70, 75, 80])
    ax.set_rlabel_position(30)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", labelsize=6.5)

    dense_line, = ax.plot(angles, dense, color=MHA_COLOR, linestyle="--", marker="o",
                          label="Multi-head", zorder=3, **MARKER)
    ax.fill(angles, dense, color=MHA_COLOR, alpha=0.10)

    ssog_line, = ax.plot(angles, ssog, color=SSOG_COLOR, marker="^",
                         label="Proposed", zorder=3, **MARKER)
    ax.fill(angles, ssog, color=SSOG_COLOR, alpha=0.10)

    ax.legend(handles=[ssog_line, dense_line], loc="upper right",
              bbox_to_anchor=(1.18, 1.12), frameon=False, fontsize=7.5)


def draw_size_sweep(ax, rows, resolution: int = 512):
    at_res = [r for r in rows if r["resolution"] == resolution and r["batch_size"] == 8]
    by_model = {r["model"]: r for r in at_res}

    for family, by_operator in eff.FAMILY_MODELS.items():
        color = LOCAL_FAMILY_COLORS[family]
        for operator, names in by_operator.items():
            family_rows = [by_model[n] for n in names if n in by_model]
            family_rows.sort(key=lambda row: row["parameters"])
            if not family_rows:
                continue
            xs = [row["parameters"] / 1e6 for row in family_rows]
            ys = [row["images_per_second"] for row in family_rows]
            ax.scatter(xs, ys, s=26, color=color, marker=eff.OPERATOR_MARKERS[operator],
                      edgecolors="white", linewidths=0.8, zorder=3)

    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set(xlabel="Parameters", ylabel="Throughput (images/s)")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}M"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs="all"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(length=2.5, pad=2)
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, color=color,
               label=LOCAL_FAMILY_LABELS[family])
        for family, color in LOCAL_FAMILY_COLORS.items()
    ]
    operator_handles = [
        Line2D([0], [0], marker=eff.OPERATOR_MARKERS[operator], linestyle="", markersize=5,
               color="black",
              label=LOCAL_OPERATOR_LABELS[operator])
        for operator in LOCAL_OPERATOR_ORDER
    ]
    legend_style = dict(frameon=False, loc="upper right", fontsize=7.5, title_fontsize=7.5,
                        handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    legend1 = ax.legend(handles=operator_handles, title="Attention operator", **legend_style)
    ax.add_artist(legend1)
    ax.legend(handles=family_handles, title="Architecture setting",
              bbox_to_anchor=(1.0, 0.74), **legend_style)


def main():
    configure_style()

    fig = plt.figure(figsize=(6.4, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.1)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_radar = fig.add_subplot(gs[0, 1], projection="polar")

    draw_radar(ax_radar)
    sizes = eff.load_rows(eff.DEFAULT_SIZES)
    draw_size_sweep(ax_scatter, sizes, resolution=1024)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.canvas.draw()

    pos_scatter = ax_scatter.get_position()
    pos_radar = ax_radar.get_position()
    title_y = max(pos_scatter.y1, pos_radar.y1) + 0.07  # clears the radar's top label
    title_size = plt.rcParams["axes.titlesize"]
    fig.text((pos_scatter.x0 + pos_scatter.x1) / 2, title_y,
              "Architecture scaling at 1024", ha="center", va="bottom",
              fontsize=title_size)
    fig.text((pos_radar.x0 + pos_radar.x1) / 2, title_y,
              "CIFAR-100 top-1 accuracy", ha="center", va="bottom",
              fontsize=title_size)

    output_dir = eff.DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"visual_abstract.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote visual_abstract.pdf / .png")


if __name__ == "__main__":
    main()
