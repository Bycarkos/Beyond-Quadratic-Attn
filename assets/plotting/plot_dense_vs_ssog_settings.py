#!/usr/bin/env python3
"""Dense attention vs. SSoG across all three architecture settings.

Grouped bar chart, one panel per architecture setting (continuous-depth
ODEViT, discrete ViT, looped/recurrent bViT), bars grouped by trained width.
Accuracy is the bar height; batch-1 latency is printed above each bar rather
than encoded as a second axis.

Data provenance:
  - Latency: experiments/SSOG/efficiency/paired_settings/resolution_scaling.csv,
    a single hardware-matched NVIDIA L40S sweep (job 13104, 2026-09-15, the
    rerun that added the medium/dim576 rows -- supersedes job 13097),
    batch_size=1, 224px, fp16, randomly initialized weights. The looped/bViT
    setting has no dense-attention latency row in that sweep yet (only the
    looped-SSOG variant was benchmarked), so its dense bars show accuracy
    with no latency label until that benchmark is run.
  - Accuracy: real trained CIFAR-100 checkpoints (final epoch 499), sourced
    from ablation_report.md sections 4.1, 4.3, 4.4, 5.1, 11.10, cross-checked
    against each run's W&B history directly (job 13101 for the looped/bViT
    dim384 dense point, added 2026-09-15).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("experiments/SSOG/figures/efficiency")

COLOR_SSOG = "#0072B2"
COLOR_DENSE = "#8C8C8C"
TEXT_GRAY = "#555555"

# Each setting: (title, [(width_label, dense_acc_or_None, dense_ms_or_None,
#                          ssog_acc, ssog_ms, ssog_acc_err_or_None,
#                          note_or_None), ...])
SETTINGS = [
    (
        "Continuous-depth (ODEViT)",
        [
            ("dim192", 64.65, 12.45, 38.93, 12.28, None, None),
            ("dim384", 69.15, 12.64, 71.37, 12.48, None, None),
            ("dim576", 69.36, 12.53, 73.48, 12.47, None, None),
            ("dim768", None, None, 71.48, 12.33, 2.17, "dense still\ntraining"),
        ],
    ),
    (
        "Discrete (12 blocks)",
        [
            ("dim384", 74.51, 5.38, 77.56, 12.22, None, None),
            ("dim576", 74.20, 5.45, 78.18, 12.32, None, None),
            ("dim768", 74.67, 5.41, 78.63, 12.34, None, None),
        ],
    ),
    (
        "Looped (1 shared block)",
        [
            ("dim384", 72.04, None, 77.46, 12.24, None, None),
            ("dim576", None, None, 79.38, 12.45, None, "dense still\ntraining"),
        ],
    ),
]


def configure_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.6,
        "grid.alpha": 0.4, "axes.axisbelow": True,
    })


def main() -> None:
    configure_style()
    widths_per_panel = [len(rows) for _, rows in SETTINGS]
    fig, axes = plt.subplots(
        1, 3, figsize=(9.6, 3.6), sharey=True,
        gridspec_kw={"width_ratios": widths_per_panel, "wspace": 0.08},
    )

    bar_w = 0.36
    pitch = 1.0
    label_bbox = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
    for ax, (title, rows) in zip(axes, SETTINGS):
        positions = [i * pitch for i in range(len(rows))]
        for pos, (width, d_acc, d_ms, s_acc, s_ms, s_err, note) in zip(positions, rows):
            if d_acc is not None:
                ax.bar(pos - bar_w / 2, d_acc, bar_w, color=COLOR_DENSE, zorder=3)
                if d_ms is not None:
                    ax.text(pos - bar_w / 2, d_acc + 1.2, f"{d_ms:.1f} ms", ha="center",
                            fontsize=6.3, color=TEXT_GRAY, zorder=4, bbox=label_bbox)
            ax.bar(pos + bar_w / 2, s_acc, bar_w, color=COLOR_SSOG, zorder=3,
                   yerr=s_err, error_kw=dict(elinewidth=1.1, capsize=2.5, ecolor="#04405e"))
            ax.text(pos + bar_w / 2, s_acc + (s_err or 0) + 1.2, f"{s_ms:.1f} ms",
                    ha="center", fontsize=6.3, color=TEXT_GRAY, zorder=4, bbox=label_bbox)
            if note:
                gap_x = pos - bar_w / 2 if d_acc is None else pos + bar_w / 2
                gap_y = 6 if d_acc is None else s_acc + (s_err or 0) + 6.5
                ax.text(gap_x, gap_y, note, ha="center", va="bottom",
                        fontsize=6.2, color="#999999", style="italic")
        ax.set_xticks(positions)
        ax.set_xticklabels([r[0] for r in rows], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.65, (len(rows) - 1) * pitch + 0.65)

    axes[0].set_ylabel("CIFAR-100 final Top-1 accuracy (%)")
    axes[0].set_ylim(0, 92)

    fig.suptitle("Dense attention vs. SSoG, by architecture setting", fontsize=11, y=1.03)
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", color=COLOR_DENSE, markersize=9, label="Dense attention"),
        Line2D([0], [0], marker="s", linestyle="none", color=COLOR_SSOG, markersize=9, label="SSoG (ours)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0), fontsize=8.5)

    fig.text(
        0.01, -0.05,
        "Labels above each bar: batch-1 latency (ms, NVIDIA L40S, FP16, 224px, randomly initialized weights, single run).\n"
        "Accuracy is a single training seed per bar, except continuous-depth SSoG at dim768 (three-seed mean $\\pm$ std).",
        fontsize=6.5, color="#666666", va="top",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"dense_vs_ssog_settings.{ext}")
    print(f"Wrote {OUTPUT_DIR / 'dense_vs_ssog_settings.pdf'}")


if __name__ == "__main__":
    main()
