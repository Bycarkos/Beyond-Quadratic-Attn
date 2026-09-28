#!/usr/bin/env python3
"""Generate publication-ready SSOG efficiency figures from benchmark CSVs.

Two independent studies feed this script:
  - The flagship 3-model RTX 3090 study (SSOG-ODEViT vs. dense ODEViT vs.
    ViT-B/16 at dim768, resolutions up to 1024, batches up to 128), read from
    resolution_extended/ and batch_scaling_512/. These feed the original
    efficiency_overview / efficiency_normalized_scaling / efficiency_pareto_512
    / efficiency_speedup figures described in detail in the paper's main text
    (Section "Resolution and batch scaling").
  - The broader 6-model A40 study (continuous-depth ODEViT, discrete ViT, and
    looped bViT, each with dense attention or SSoG, Tiny/Small/Base/Large),
    read from transformer_sizes/. These feed the "_all_settings" figures plus
    the two efficiency_model_sizes_{512,1024} architecture-scaling figures,
    described as a secondary, broader analysis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENDED = ROOT / "experiments/SSOG/efficiency/resolution_extended/resolution_scaling.csv"
DEFAULT_BATCH = ROOT / "experiments/SSOG/efficiency/batch_scaling_512/resolution_scaling.csv"
DEFAULT_SIZES = ROOT / "experiments/SSOG/efficiency/transformer_sizes/resolution_scaling.csv"
DEFAULT_OUTPUT = ROOT / "experiments/SSOG/figures/efficiency"

# --- Flagship 3-model RTX 3090 study (SSOG-ODEViT / ODEViT / ViT-B/16) ---
LABELS = {"ssog": "SSOG (ours)", "odevit": "ODEViT", "vit": "ViT-B/16"}
COLORS = {"ssog": "#0072B2", "odevit": "#D55E00", "vit": "#4D4D4D"}
MARKERS = {"ssog": "o", "odevit": "s", "vit": "^"}
LINESTYLES = {"ssog": "-", "odevit": "--", "vit": "-."}

# --- Broader 6-model A40 study: color = architecture setting, marker shape
# = attention operator, used consistently across every "_all_settings" and
# "model_sizes" figure so the encoding never has to be relearned. ---
FAMILY_COLORS = {"odevit": "#E69F00", "vit": "#0072B2", "bvit": "#CC79A7"}
FAMILY_LABELS = {"odevit": "ODEViT", "vit": "ViT", "bvit": "bViT"}
OPERATOR_MARKERS = {"dense": "o", "ssog": "^"}
OPERATOR_LABELS = {"dense": "Dense", "ssog": "SSoG"}
OPERATOR_LINESTYLES = {"dense": "-", "ssog": "--"}
# Each family's dense and SSoG model names, in increasing size order
# (tiny, small, base, large); index 2 ("base") is the dim768-equivalent
# point used by every "_all_settings" figure (all except the size sweep).
FAMILY_MODELS = {
    "odevit": {
        "dense": ["odevit_tiny_dense", "odevit_small_dense", "odevit_base_dense",
                 "odevit_large_dense"],
        "ssog": ["odevit_tiny_ssog", "odevit_small_ssog", "odevit_base_ssog",
                "odevit_large_ssog"],
    },
    "vit": {
        "dense": ["vit_tiny", "vit_small", "vit_base", "vit_large"],
        "ssog": ["vit_tiny_ssog", "vit_small_ssog", "vit_base_ssog", "vit_large_ssog"],
    },
    "bvit": {
        "dense": ["bvit_tiny_dense", "bvit_small_dense", "bvit_base_dense",
                 "bvit_large_dense"],
        "ssog": ["bvit_tiny_ssog", "bvit_small_ssog", "bvit_base_ssog", "bvit_large_ssog"],
    },
}
SIZE_RESOLUTIONS = [224, 384, 512, 1024]


def iter_base_models():
    """Yield (family, operator, base-size model name) for all six types."""
    for family, by_operator in FAMILY_MODELS.items():
        for operator, names in by_operator.items():
            yield family, operator, names[2]


def configure_style() -> str:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
        style = "SciencePlots: science + no-latex + grid"
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
        style = "matplotlib fallback (install scienceplots for SciencePlots style)"
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return style


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"No successful benchmark rows found in {path}")
    for row in rows:
        for key in (
            "batch_size",
            "images_per_second",
            "latency_mean_ms",
            "parameters",
            "peak_allocated_mib",
            "resolution",
        ):
            row[key] = float(row[key])
        # FLOPs are only profiled at the batch size(s) in
        # profile_flops_batch_sizes (default: the first batch size), so this
        # field is legitimately empty for other rows; it isn't used by any
        # plot in this file, so keep it as-is rather than failing the load.
        if row.get("profiler_gflops_per_image"):
            row["profiler_gflops_per_image"] = float(row["profiler_gflops_per_image"])
    return rows


def save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


# =====================================================================
# Flagship 3-model RTX 3090 study
# =====================================================================

def select(rows: list[dict], model: str, batch: int) -> list[dict]:
    return sorted(
        (r for r in rows if r["model"] == model and r["batch_size"] == batch),
        key=lambda r: r["resolution"],
    )


def draw_series(ax, rows: list[dict], metric: str, model: str, label: str | None = None):
    ax.plot(
        [r["resolution"] for r in rows],
        [r[metric] for r in rows],
        color=COLORS[model],
        marker=MARKERS[model],
        linestyle=LINESTYLES[model],
        label=label or LABELS[model],
        zorder=3 if model == "ssog" else 2,
    )


def plot_overview(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    panels = (
        (8, "images_per_second", "Batch 8 throughput", "Throughput (images/s)", 512),
        (32, "images_per_second", "Batch 32 throughput", "Throughput (images/s)", 640),
        (32, "peak_allocated_mib", "Batch 32 memory (lower is better)", "Peak allocated memory (MiB)", None),
    )
    for ax, (batch, metric, title, ylabel, crossover) in zip(axes, panels):
        for model in LABELS:
            draw_series(ax, select(rows, model, batch), metric, model)
        if crossover is not None:
            ax.axvline(crossover, color=COLORS["ssog"], linestyle=":", linewidth=1.1)
            ax.text(crossover + 12, 0.93, "SSOG leads", color=COLORS["ssog"],
                    transform=ax.get_xaxis_transform(), fontsize=7, va="top")
        ax.set(title=title, xlabel="Input resolution", ylabel=ylabel)
        ax.set_xticks([224, 384, 512, 640, 768, 1024])
        ax.tick_params(axis="x", rotation=35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.79, wspace=0.36)
    save(fig, output_dir, "efficiency_overview")


def plot_normalized(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for model in LABELS:
        model_rows = select(rows, model, 32)
        throughput0 = model_rows[0]["images_per_second"]
        memory0 = model_rows[0]["peak_allocated_mib"]
        axes[0].plot(
            [r["resolution"] for r in model_rows],
            [r["images_per_second"] / throughput0 for r in model_rows],
            color=COLORS[model], marker=MARKERS[model], linestyle=LINESTYLES[model],
            label=LABELS[model],
        )
        axes[1].plot(
            [r["resolution"] for r in model_rows],
            [r["peak_allocated_mib"] / memory0 for r in model_rows],
            color=COLORS[model], marker=MARKERS[model], linestyle=LINESTYLES[model],
            label=LABELS[model],
        )
    axes[0].set(title="Throughput retention", ylabel="Relative to 224", xlabel="Input resolution")
    axes[1].set(title="Memory growth", ylabel="Relative to 224", xlabel="Input resolution")
    for ax in axes:
        ax.set_xticks([224, 384, 512, 640, 768, 1024])
        ax.tick_params(axis="x", rotation=35)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.79, wspace=0.34)
    save(fig, output_dir, "efficiency_normalized_scaling")


def plot_pareto(rows: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    label_offsets = {
        "ssog": (5, 5),
        "odevit": (5, -10),
        "vit": (-24, 5),
    }
    for model in LABELS:
        model_rows = sorted((r for r in rows if r["model"] == model), key=lambda r: r["batch_size"])
        x = [r["peak_allocated_mib"] for r in model_rows]
        y = [r["images_per_second"] for r in model_rows]
        ax.plot(x, y, color=COLORS[model], marker=MARKERS[model], linestyle=LINESTYLES[model], label=LABELS[model])
        for r in model_rows:
            if r["batch_size"] in (1, 8, 32, 128):
                ax.annotate(f"B{int(r['batch_size'])}", (r["peak_allocated_mib"], r["images_per_second"]),
                            xytext=label_offsets[model], textcoords="offset points", fontsize=7)
    ax.set(xlabel="Peak allocated memory (MiB)", ylabel="Throughput (images/s)",
           title="Throughput--memory trade-off at 512")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    save(fig, output_dir, "efficiency_pareto_512")


def plot_speedup(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, baseline in zip(axes, ("odevit", "vit")):
        for batch, marker in ((8, "o"), (32, "s")):
            ours = {r["resolution"]: r for r in select(rows, "ssog", batch)}
            base = {r["resolution"]: r for r in select(rows, baseline, batch)}
            resolutions = sorted(ours.keys() & base.keys())
            speedup = [ours[x]["images_per_second"] / base[x]["images_per_second"] for x in resolutions]
            ax.plot(resolutions, speedup, marker=marker, label=f"Batch {batch}")
        ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set(title=f"SSOG / {LABELS[baseline]}", xlabel="Input resolution")
        ax.text(0.03, 0.95, "SSOG faster above 1.0", color=COLORS["ssog"],
                transform=ax.transAxes, fontsize=7, va="top")
        ax.set_xticks([224, 384, 512, 640, 768, 1024])
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Throughput speedup ($\\times$)")
    axes[1].legend(frameon=False)
    fig.subplots_adjust(wspace=0.12)
    save(fig, output_dir, "efficiency_speedup")


# =====================================================================
# Broader 6-model A40 study (continuous ODEViT / discrete ViT / looped
# bViT, each dense or SSoG)
# =====================================================================

def select_resolution(rows: list[dict], model: str, batch: int) -> list[dict]:
    return sorted(
        (r for r in rows if r["model"] == model and r["batch_size"] == batch),
        key=lambda r: r["resolution"],
    )


def draw_base_series(ax, rows: list[dict], metric: str, batch: int):
    """Plot every family/operator's base-size series (throughput or memory
    vs. resolution) with the shared color=family, shape=operator convention."""
    for family, operator, model in iter_base_models():
        model_rows = select_resolution(rows, model, batch)
        if not model_rows:
            continue
        ax.plot(
            [r["resolution"] for r in model_rows],
            [r[metric] for r in model_rows],
            color=FAMILY_COLORS[family], marker=OPERATOR_MARKERS[operator],
            linestyle=OPERATOR_LINESTYLES[operator],
            label=f"{FAMILY_LABELS[family]} ({OPERATOR_LABELS[operator]})",
        )


def plot_overview_all_settings(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    panels = (
        (1, "images_per_second", "Batch 1 throughput", "Throughput (images/s)"),
        (8, "images_per_second", "Batch 8 throughput", "Throughput (images/s)"),
        (8, "peak_allocated_mib", "Batch 8 memory (lower is better)", "Peak allocated memory (MiB)"),
    )
    for ax, (batch, metric, title, ylabel) in zip(axes, panels):
        draw_base_series(ax, rows, metric, batch)
        ax.set(title=title, xlabel="Input resolution", ylabel=ylabel)
        ax.set_xscale("log")
        ax.set_xticks(SIZE_RESOLUTIONS)
        ax.set_xticklabels(SIZE_RESOLUTIONS, rotation=35)
        ax.minorticks_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.2)
    fig.subplots_adjust(top=0.7, wspace=0.38)
    save(fig, output_dir, "efficiency_overview_all_settings")


def plot_normalized_all_settings(rows: list[dict], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for family, operator, model in iter_base_models():
        model_rows = select_resolution(rows, model, 8)
        if not model_rows:
            continue
        throughput0 = model_rows[0]["images_per_second"]
        memory0 = model_rows[0]["peak_allocated_mib"]
        style = dict(color=FAMILY_COLORS[family], marker=OPERATOR_MARKERS[operator],
                    linestyle=OPERATOR_LINESTYLES[operator],
                    label=f"{FAMILY_LABELS[family]} ({OPERATOR_LABELS[operator]})")
        axes[0].plot([r["resolution"] for r in model_rows],
                    [r["images_per_second"] / throughput0 for r in model_rows], **style)
        axes[1].plot([r["resolution"] for r in model_rows],
                    [r["peak_allocated_mib"] / memory0 for r in model_rows], **style)
    axes[0].set(title="Throughput retention", ylabel="Relative to 224", xlabel="Input resolution")
    axes[1].set(title="Memory growth", ylabel="Relative to 224", xlabel="Input resolution")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(SIZE_RESOLUTIONS)
        ax.set_xticklabels(SIZE_RESOLUTIONS, rotation=35)
        ax.minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
              frameon=False, fontsize=7.2)
    fig.subplots_adjust(top=0.7, wspace=0.34)
    save(fig, output_dir, "efficiency_normalized_all_settings")


def plot_pareto_all_settings(rows: list[dict], output_dir: Path):
    """Throughput vs. memory as resolution grows (batch 8), one point per
    resolution per model -- a resolution-indexed frontier rather than the
    batch-indexed one in the flagship 3-model study, since transformer_sizes
    only benchmarks batch 1 and 8, not a full batch sweep."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for family, operator, model in iter_base_models():
        model_rows = select_resolution(rows, model, 8)
        if not model_rows:
            continue
        ax.plot([r["peak_allocated_mib"] for r in model_rows],
               [r["images_per_second"] for r in model_rows],
               color=FAMILY_COLORS[family], marker=OPERATOR_MARKERS[operator],
               linestyle=OPERATOR_LINESTYLES[operator],
               label=f"{FAMILY_LABELS[family]} ({OPERATOR_LABELS[operator]})")
    ax.set(xlabel="Peak allocated memory (MiB)", ylabel="Throughput (images/s)",
           title="Throughput--memory trade-off across resolution (batch 8)")
    ax.set_xscale("log")
    memory_ticks = [200, 500, 1000, 2000, 4000]
    ax.set_xticks(memory_ticks)
    ax.set_xticklabels([str(t) for t in memory_ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=7.2, ncol=2)
    save(fig, output_dir, "efficiency_pareto_all_settings")


def plot_speedup_all_settings(rows: list[dict], output_dir: Path):
    """SSoG/Dense speedup within each matched architecture setting (not
    across settings, since each setting now has its own dense baseline)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    for ax, family in zip(axes, FAMILY_COLORS):
        for batch, marker in ((1, "o"), (8, "^")):
            ssog_name = FAMILY_MODELS[family]["ssog"][2]
            dense_name = FAMILY_MODELS[family]["dense"][2]
            ours = {r["resolution"]: r for r in select_resolution(rows, ssog_name, batch)}
            base = {r["resolution"]: r for r in select_resolution(rows, dense_name, batch)}
            resolutions = sorted(ours.keys() & base.keys())
            speedup = [ours[x]["images_per_second"] / base[x]["images_per_second"] for x in resolutions]
            ax.plot(resolutions, speedup, marker=marker, color=FAMILY_COLORS[family],
                   linestyle="-" if batch == 1 else "--", label=f"Batch {batch}")
        ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set(title=FAMILY_LABELS[family], xlabel="Input resolution")
        ax.set_xscale("log")
        ax.set_xticks(SIZE_RESOLUTIONS)
        ax.set_xticklabels(SIZE_RESOLUTIONS, rotation=35)
        ax.minorticks_off()
    axes[0].set_ylabel("SSoG / Dense throughput ratio")
    axes[-1].legend(frameon=False, fontsize=7.5)
    fig.suptitle("SSoG speedup over dense attention, by architecture setting", fontsize=10)
    fig.subplots_adjust(top=0.8, wspace=0.12)
    save(fig, output_dir, "efficiency_speedup_all_settings")


def plot_size_sweep(rows: list[dict], output_dir: Path, resolution: int = 512):
    at_res = [r for r in rows if r["resolution"] == resolution and r["batch_size"] == 8]
    by_model = {r["model"]: r for r in at_res}

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for family, by_operator in FAMILY_MODELS.items():
        color = FAMILY_COLORS[family]
        for operator, names in by_operator.items():
            family_rows = [by_model[n] for n in names if n in by_model]
            family_rows.sort(key=lambda row: row["parameters"])
            if not family_rows:
                continue
            xs = [row["parameters"] / 1e6 for row in family_rows]
            ys = [row["images_per_second"] for row in family_rows]
            ax.scatter(xs, ys, s=90, color=color, marker=OPERATOR_MARKERS[operator],
                      edgecolors="white", linewidths=1.1, zorder=3)

    ax.set(xlabel="Parameters (millions, log scale)", ylabel="Throughput (images/s)",
           title=f"Architecture scaling at {resolution}")
    ax.set_xscale("log")
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=FAMILY_LABELS[family])
        for family, color in FAMILY_COLORS.items()
    ]
    operator_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="black",
              label=OPERATOR_LABELS[operator])
        for operator, marker in OPERATOR_MARKERS.items()
    ]
    legend1 = ax.legend(handles=family_handles, frameon=False, loc="upper right",
                        title="Architecture setting")
    ax.add_artist(legend1)
    ax.legend(handles=operator_handles, frameon=False, loc="upper right",
             title="Attention operator", bbox_to_anchor=(1.0, 0.72))
    save(fig, output_dir, f"efficiency_model_sizes_{resolution}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--sizes", type=Path, default=DEFAULT_SIZES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    style = configure_style()
    extended = load_rows(args.extended)
    batch = load_rows(args.batch)
    sizes = load_rows(args.sizes)

    plot_overview(extended, args.output_dir)
    plot_normalized(extended, args.output_dir)
    plot_pareto(batch, args.output_dir)
    plot_speedup(extended, args.output_dir)

    plot_overview_all_settings(sizes, args.output_dir)
    plot_normalized_all_settings(sizes, args.output_dir)
    plot_pareto_all_settings(sizes, args.output_dir)
    plot_speedup_all_settings(sizes, args.output_dir)
    plot_size_sweep(sizes, args.output_dir, resolution=512)
    plot_size_sweep(sizes, args.output_dir, resolution=1024)

    print(f"Generated 10 PDF and 10 PNG figures in {args.output_dir}")
    print(f"Style: {style}")


if __name__ == "__main__":
    main()
