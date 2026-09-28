#!/usr/bin/env python3
"""SSOG improvement over dense (%) vs resolution, one single-axis panel per
architecture family. Both series are oriented so that positive = better for
SSOG: throughput delta (SSOG faster than dense), memory SAVED (dense
memory minus SSOG memory, i.e. the sign of the raw memory delta is
flipped) and FLOPs SAVED (whole-model forward FLOPs, from assets/benchmarking/count_flops.py).
A dashed line at 0% marks dense parity. Measured data comes exclusively
from the six fully isolated per-model benchmark re-runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ISOLATED_CSVS = {
    ("odevit", "dense"): ROOT / "experiments/SSOG/efficiency/odevit_dense_isolated/resolution_scaling.csv",
    ("odevit", "ssog"): ROOT / "experiments/SSOG/efficiency/odevit_ssog_isolated/resolution_scaling.csv",
    ("vit", "dense"): ROOT / "experiments/SSOG/efficiency/vit_dense_isolated/resolution_scaling.csv",
    ("vit", "ssog"): ROOT / "experiments/SSOG/efficiency/vit_ssog_isolated/resolution_scaling.csv",
    ("bvit", "dense"): ROOT / "experiments/SSOG/efficiency/bvit_dense_isolated/resolution_scaling.csv",
    ("bvit", "ssog"): ROOT / "experiments/SSOG/efficiency/bvit_ssog_isolated/resolution_scaling.csv",
}
FLOPS_CSV = ROOT / "experiments/SSOG/efficiency/flops/flops.csv"
DEFAULT_OUTPUT = ROOT / "experiments/SSOG/figures/efficiency"

RESOLUTIONS = [224, 384, 512, 640, 768, 1024]
BATCHES = [1, 8, 32, 64, 128, 256, 512]
INK = "#2b2b2b"

FAMILY_ORDER = ["odevit", "vit", "bvit"]
FAMILY_TITLES = {
    "odevit": "Continuous (ODEViT)",
    "vit": "Discrete (ViT-B/16)",
    "bvit": "Looped (bViT)",
}

MHA_COLOR, SSOG_COLOR = "#D55E00", "#0072B2"  # Okabe-Ito, as in efficiency_delta_by_resolution
INK, MUTED, GRID = "black", "#777777", "#ececec"

THROUGHPUT_COLOR = "#0072B2"  # Okabe-Ito blue
MEMORY_COLOR = "#D55E00"      # Okabe-Ito vermillion
FLOPS_COLOR = "black"         # theoretical count, kept neutral vs. the measured series


# Style prototype: the horizontal ablation figure. Imported so both figures
# share one set of rcParams and cannot drift apart.
from assets.plotting.plot_ablation_horizontal import GRID, MUTED, configure_style  # noqa: E402


def load():
    throughput, memory = {}, {}
    for (fam, mixer), path in ISOLATED_CSVS.items():
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "ok":
                    continue
                key = (fam, mixer, int(float(row["resolution"])), int(float(row["batch_size"])))
                throughput[key] = float(row["images_per_second"])
                memory[key] = float(row["peak_allocated_mib"])
    return throughput, memory


def load_flops_saved():
    """% forward FLOPs saved by SSoG vs dense per resolution, for the whole
    model and for the attention modules. The three settings agree to within
    0.3 GFLOPs (each applies a dim768 block 12 times), so the discrete
    setting's counts stand for all of them."""
    flops = {}
    with open(FLOPS_CSV) as handle:
        for row in csv.DictReader(handle):
            if row["family"] == "vit":
                flops[row["mixer"], int(row["resolution"])] = (
                    float(row["total_gflops"]), float(row["attention_gflops"]))
    return {part: {res: -pct(flops["dense", res][i], flops["ssog", res][i]) for res in RESOLUTIONS}
            for i, part in enumerate(("model", "attention"))}


def pct(a, b):
    return (b - a) / a * 100


def collapse_by_resolution(throughput, memory):
    thr_by_res = {fam: {} for fam in FAMILY_ORDER}
    mem_saved_by_res = {fam: {} for fam in FAMILY_ORDER}
    for fam in FAMILY_ORDER:
        for res in RESOLUTIONS:
            thr_vals, mem_vals = [], []
            for b in BATCHES:
                dt = throughput.get((fam, "dense", res, b))
                st = throughput.get((fam, "ssog", res, b))
                dm = memory.get((fam, "dense", res, b))
                sm = memory.get((fam, "ssog", res, b))
                if dt is None or st is None or dm is None or sm is None:
                    continue
                thr_vals.append(pct(dt, st))          # + = SSOG faster
                mem_vals.append(-pct(dm, sm))         # + = SSOG uses less memory
            if thr_vals:
                thr_by_res[fam][res] = sum(thr_vals) / len(thr_vals)
                mem_saved_by_res[fam][res] = sum(mem_vals) / len(mem_vals)
    return thr_by_res, mem_saved_by_res


def style_axis(ax, res_list, title):
    xs = list(range(len(res_list)))  # evenly spaced, as in the ablation prototype
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r) for r in res_list])
    ax.set_xlim(-0.35, xs[-1] + 0.35)
    ax.set_xlabel("Resolution")
    ax.set_title(title, pad=5)
    ax.tick_params(length=2.5, pad=2)
    return xs


MARKER_STYLE = dict(markersize=4.5, markeredgecolor="white", markeredgewidth=1.0, zorder=3)


def plot_flops(ax, flops_saved):
    """Fourth panel: arithmetic cost, shared by all three settings."""
    xs = style_axis(ax, RESOLUTIONS, "FLOPs (all settings)")
    model = [flops_saved["model"][r] for r in RESOLUTIONS]
    attention = [flops_saved["attention"][r] for r in RESOLUTIONS]
    ax.plot(xs, attention, color=FLOPS_COLOR, marker="s", linestyle="-", **MARKER_STYLE)
    ax.plot(xs, model, color=FLOPS_COLOR, marker="s", linestyle=":", markerfacecolor="white",
            markersize=3.8, markeredgecolor=FLOPS_COLOR, markeredgewidth=0.9, zorder=3)
    ax.annotate("Attention", (xs[1], attention[1]), xytext=(0, 5), textcoords="offset points",
                color=FLOPS_COLOR, fontsize=8, ha="center", va="bottom")
    ax.annotate("Whole model", (xs[1], model[1]), xytext=(0, 7), textcoords="offset points",
                color=FLOPS_COLOR, fontsize=8, ha="center", va="bottom")
    ax.set_ylim(-5, 90)


def plot_one(ax, fam, thr_by_res, mem_by_res, show_parity_label: bool,
             show_series_labels: bool):
    res_list = sorted(thr_by_res[fam])
    xs = style_axis(ax, res_list, FAMILY_TITLES[fam])
    thr_ys = [thr_by_res[fam][res] for res in res_list]
    mem_ys = [mem_by_res[fam][res] for res in res_list]

    if show_parity_label:
        ax.text(xs[2], 1.5, "dense parity", color=MUTED, fontsize=7,
                style="italic", ha="center", va="bottom")

    ax.plot(xs, thr_ys, color=THROUGHPUT_COLOR, marker="o", linestyle="-", **MARKER_STYLE)
    ax.plot(xs, mem_ys, color=MEMORY_COLOR, marker="^", linestyle="--", **MARKER_STYLE)

    if show_series_labels:  # direct labels once, in the panel where the lines are furthest apart
        ax.annotate("Memory saved", (xs[2], mem_ys[2]), xytext=(0, 5), textcoords="offset points",
                    color=MEMORY_COLOR, fontsize=8, ha="center", va="bottom")
        ax.annotate("Throughput", (xs[1], thr_ys[1]), xytext=(4, -6), textcoords="offset points",
                    color=THROUGHPUT_COLOR, fontsize=8, ha="left", va="top")


def plot_all_families(thr_by_res, mem_by_res, flops_saved, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.1), sharey=False)
    for i, (ax, fam) in enumerate(zip(axes, FAMILY_ORDER)):
        plot_one(ax, fam, thr_by_res, mem_by_res, show_parity_label=(i == 0),
                 show_series_labels=(i == 1))
    plot_flops(axes[3], flops_saved)
    axes[0].set_ylabel("Improvement over dense (%)")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "efficiency_delta_by_resolution"
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .png")


def main():
    configure_style()
    throughput, memory = load()
    thr_by_res, mem_by_res = collapse_by_resolution(throughput, memory)
    plot_all_families(thr_by_res, mem_by_res, load_flops_saved(), DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
