#!/usr/bin/env python3
"""Grid version of plot_efficiency_dual_axis.py: one dual-axis
throughput/memory-vs-resolution panel per (batch size, family) pair, all in
a single figure (rows = batch size, columns = family) for the appendix.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from assets.plotting import plot_efficiency_dual_axis as dual

BATCHES = [8, 32, 64, 128, 256, 512]

INK, MUTED, GRID = "black", "#777777", "#ececec"

def plot_grid(idx: dict, batches: list[int], output_dir, oom_keys: set = frozenset()):
    n_rows = len(dual.FAMILY_ORDER)
    n_cols = len(batches)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows))

    handles = None
    oom_handles_by_label = {}
    for row, family in enumerate(dual.FAMILY_ORDER):
        for col, batch in enumerate(batches):
            ax = axes[row, col]
            is_first_col = col == 0
            is_last_col = col == n_cols - 1
            h, oom_h = dual.plot_one(ax, idx, family, batch, is_first=is_first_col,
                                      is_last=is_last_col, oom_keys=oom_keys)
            if h:
                handles = h
            for oh in oom_h:
                oom_handles_by_label[oh.get_label()] = oh

            if row == 0:
                ax.set_title(f"Batch {batch}")
            else:
                ax.set_title("")

            if row != n_rows - 1:
                ax.set_xlabel("")

            if is_first_col:
                current = ax.get_ylabel()
                ax.set_ylabel(f"{dual.FAMILY_TITLES[family]}\n{current}", color=dual.THROUGHPUT_COLOR)

    all_handles = handles + list(oom_handles_by_label.values())
    fig.legend(
        handles=all_handles, loc="upper center", ncol=len(all_handles), framealpha=0.9,
        bbox_to_anchor=(0.5, 1.0 + 0.4 / (3.0 * n_rows)),
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "efficiency_dual_axis_grid_all_batches"
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .png")


def main():
    dual.configure_style()
    rows = dual.load_rows(dual.DEFAULT_CSVS)
    idx = dual.index_rows(rows)
    oom_keys = dual.load_oom_keys(dual.DEFAULT_CSVS)
    plot_grid(idx, BATCHES, dual.DEFAULT_OUTPUT, oom_keys=oom_keys)


if __name__ == "__main__":
    main()
