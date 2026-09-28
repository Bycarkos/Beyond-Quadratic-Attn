#!/usr/bin/env python3
"""Per-family dual-axis efficiency plots: throughput (left) vs peak memory
(right) against BATCH SIZE at a fixed resolution, dense vs SSOG, for each of
the three architecture settings (continuous/ODEViT, discrete/ViT,
looped/bViT). Complements plot_efficiency_dual_axis.py, which instead fixes
batch and sweeps resolution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSVS = [
    ROOT / "experiments/SSOG/efficiency/paired_base_a40/resolution_scaling.csv",
    ROOT / "experiments/SSOG/efficiency/paired_base_a40_batches/resolution_scaling.csv",
]
DEFAULT_OUTPUT = ROOT / "experiments/SSOG/figures/efficiency"
INK, MUTED, GRID = "black", "#777777", "#ececec"
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]

FAMILY_ORDER = ["odevit", "vit", "bvit"]
FAMILY_TITLES = {
    "odevit": "Continuous-depth (ODEViT)",
    "vit": "Discrete (ViT-B/16)",
    "bvit": "Looped (bViT)",
}

OPERATOR_LABELS = {"dense": "Dense", "ssog": "SSOG"}
OPERATOR_MARKERS = {"dense": "o", "ssog": "^"}
OPERATOR_LINESTYLES = {"dense": "-", "ssog": "--"}

THROUGHPUT_COLOR = "#0072B2"  # Okabe-Ito blue
MEMORY_COLOR = "#CC79A7"      # Okabe-Ito reddish purple

OOM_MARKERS = {"dense": "x", "ssog": "+"}
OOM_COLORS = {"dense": "#E69F00", "ssog": "#D55E00"}  # Okabe-Ito orange / vermillion


def configure_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
        
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "axes.titleweight": "normal", "axes.titlepad": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows = [row for row in rows if row.get("status") == "ok"]
    for row in rows:
        row["batch_size"] = int(float(row["batch_size"]))
        row["resolution"] = int(float(row["resolution"]))
        row["images_per_second"] = float(row["images_per_second"])
        row["peak_allocated_mib"] = float(row["peak_allocated_mib"])
    return rows


def index_rows(rows: list[dict]) -> dict:
    idx = {}
    for row in rows:
        key = (row["model_family"], row["model_mixer"], row["resolution"], row["batch_size"])
        idx[key] = row
    return idx


def load_oom_keys(paths: list[Path]) -> set:
    oom = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "oom":
                    oom.add((
                        row["model_family"], row["model_mixer"],
                        int(float(row["resolution"])), int(float(row["batch_size"])),
                    ))
    return oom


def available_batches(idx: dict, family: str, operator: str, resolution: int) -> list[int]:
    return [b for b in BATCH_SIZES if (family, operator, resolution, b) in idx]


def first_oom_batch(oom_keys: set, family: str, operator: str, resolution: int, last_ok):
    for b in BATCH_SIZES:
        if last_ok is not None and b <= last_ok:
            continue
        if (family, operator, resolution, b) in oom_keys:
            return b
    return None


def plot_one(ax_throughput, idx: dict, family: str, resolution: int, is_first: bool, is_last: bool,
             oom_keys: set = frozenset()):
    ax_memory = ax_throughput.twinx()
    ax_memory.spines["right"].set_visible(True)
    ax_memory.spines["top"].set_visible(False)

    handles = []
    oom_happened = set()
    all_xticks = set()
    for operator in ("dense", "ssog"):
        batches = available_batches(idx, family, operator, resolution)
        if not batches:
            continue
        all_xticks.update(batches)
        throughput = [idx[(family, operator, resolution, b)]["images_per_second"] for b in batches]
        memory = [idx[(family, operator, resolution, b)]["peak_allocated_mib"] for b in batches]

        (line_t,) = ax_throughput.plot(
            batches, throughput,
            color=THROUGHPUT_COLOR, marker=OPERATOR_MARKERS[operator],
            linestyle=OPERATOR_LINESTYLES[operator],
            label=f"Throughput -- {OPERATOR_LABELS[operator]}",
        )
        (line_m,) = ax_memory.plot(
            batches, memory,
            color=MEMORY_COLOR, marker=OPERATOR_MARKERS[operator],
            linestyle=OPERATOR_LINESTYLES[operator],
            label=f"Memory -- {OPERATOR_LABELS[operator]}",
        )
        handles.extend([line_t, line_m])

        oom_batch = first_oom_batch(oom_keys, family, operator, resolution, batches[-1])
        if oom_batch is not None:
            all_xticks.add(oom_batch)
            oom_happened.add(operator)
            for ax, last_val in ((ax_throughput, throughput[-1]), (ax_memory, memory[-1])):
                ax.plot(oom_batch, last_val, marker=OOM_MARKERS[operator], color=OOM_COLORS[operator],
                        markersize=8, markeredgewidth=2.0, linestyle="", zorder=4)

    batches = sorted(all_xticks)
    ax_throughput.set_xscale("log", base=2)
    ax_throughput.set_xticks(batches)
    ax_throughput.set_xticklabels([str(b) for b in batches])
    ax_throughput.minorticks_off()
    ax_throughput.set_xlabel("Batch size")

    ax_throughput.tick_params(axis="y", labelcolor=THROUGHPUT_COLOR)
    ax_throughput.spines["left"].set_color(THROUGHPUT_COLOR)
    if is_first:
        ax_throughput.set_ylabel("Throughput (images/s)", color=THROUGHPUT_COLOR)

    ax_memory.tick_params(axis="y", labelcolor=MEMORY_COLOR)
    ax_memory.spines["right"].set_color(MEMORY_COLOR)
    ax_memory.grid(False)
    if is_last:
        ax_memory.set_ylabel("Peak allocated memory (MiB)", color=MEMORY_COLOR)

    ax_throughput.set_title(FAMILY_TITLES[family])
    oom_handles = [
        Line2D([], [], linestyle="", marker=OOM_MARKERS[op], color=OOM_COLORS[op],
               markersize=8, markeredgewidth=2.0, label=f"OOM ({OPERATOR_LABELS[op]})")
        for op in ("dense", "ssog") if op in oom_happened
    ]
    return handles, oom_handles


def plot_all_families(idx: dict, resolution: int, output_dir: Path, oom_keys: set = frozenset()) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    handles = None
    oom_handles_by_label = {}
    for ax, family in zip(axes, FAMILY_ORDER):
        handles, oom_handles = plot_one(
            ax, idx, family, resolution,
            is_first=(family == FAMILY_ORDER[0]),
            is_last=(family == FAMILY_ORDER[-1]),
            oom_keys=oom_keys,
        )
        for h in oom_handles:
            oom_handles_by_label[h.get_label()] = h

    all_handles = handles + list(oom_handles_by_label.values())
    fig.legend(
        handles=all_handles, loc="upper center", ncol=len(all_handles), framealpha=0.9,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"efficiency_dual_axis_by_batch_res{resolution}"
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, nargs="+", default=DEFAULT_CSVS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1024])
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    rows = load_rows(args.csv)
    idx = index_rows(rows)
    oom_keys = load_oom_keys(args.csv)
    for resolution in args.resolutions:
        plot_all_families(idx, resolution, args.output_dir, oom_keys=oom_keys)


if __name__ == "__main__":
    main()
