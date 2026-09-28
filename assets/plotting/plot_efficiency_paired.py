"""Paper-oriented plots for paired dense-versus-SSOG efficiency results."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

FAMILIES = ["odevit", "vit", "bvit", "deit"]
SIZES = ["tiny", "small", "base", "large"]
# Validated categorical palette (dataviz skill: fixed slot order, passes the
# adjacent-pair CVD/contrast checks for a 4-series line chart). Aqua/yellow sit
# below 3:1 contrast on the surface, so those two series always carry a direct
# end-of-line label rather than relying on the legend alone.
COLORS = dict(tiny="#2a78d6", small="#eb6834", base="#1baf7a", large="#eda100")
FAMILY_COLORS = dict(odevit="#2a78d6", vit="#eb6834", bvit="#1baf7a", deit="#eda100")
SIZE_MARKERS = dict(tiny="o", small="s", base="^", large="D")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def configure_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 7.5,
            "axes.edgecolor": GRIDLINE,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def end_label(ax, x, y, text, color, dx=6, dy=0):
    """Direct end-of-line label in secondary ink, with a colored key dot."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        va="center",
        ha="left" if dx >= 0 else "right",
    )


def aggregate(path):
    frame = pd.read_csv(path)
    frame = frame[frame.status == "ok"].copy()
    keys = ["model_family", "model_size", "model_mixer", "resolution", "batch_size"]
    metrics = ["images_per_second", "latency_mean_ms", "peak_allocated_mib"]
    return frame.groupby(keys, as_index=False)[metrics].agg("mean")


def paired_metric(frame, metric, operation):
    wide = frame.pivot_table(index=["model_family", "model_size", "resolution", "batch_size"],
                             columns="model_mixer", values=metric).dropna().reset_index()
    wide["value"] = operation(wide["ssog"], wide["dense"])
    return wide


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


FAMILY_TITLES = {"odevit": "ODE-ViT", "vit": "ViT", "bvit": "bViT", "deit": "DeiT"}
X_TICKS = [224, 384, 512, 768, 1024]


def panels(frame, ylabel, title, stem, output, parity_label=None):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.995)
    for ax, family in zip(axes.flat, FAMILIES):
        part_family = frame[frame.model_family == family]
        xmax = part_family.resolution.max() if not part_family.empty else X_TICKS[-1]
        for size in SIZES:
            part = part_family[part_family.model_size == size].sort_values("resolution")
            if part.empty:
                continue
            ax.plot(part.resolution, part.value, marker="o", color=COLORS[size],
                    markeredgecolor="#fcfcfb", markeredgewidth=1.2, zorder=3)
            last = part.iloc[-1]
            end_label(ax, last.resolution, last.value, size.title(), COLORS[size])
        if parity_label:
            ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
        ax.set_title(FAMILY_TITLES[family], fontsize=11, fontweight="bold", color=INK_PRIMARY)
        ax.set_xlabel("Input resolution")
        ax.set_xticks(X_TICKS)
        ax.margins(x=0.16)
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")
    if parity_label:
        axes[0, 1].annotate(parity_label, xy=(1.0, 1.0), xycoords=("axes fraction", "data"),
                             xytext=(-4, 6), textcoords="offset points", fontsize=8.5,
                             color=INK_MUTED, ha="right", va="bottom", style="italic")
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    size_handles = [plt.Line2D([], [], color=COLORS[s], marker="o", label=s.title()) for s in SIZES]
    fig.legend(handles=size_handles, title="Model size", loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.56, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, output, stem)


def throughput_resolution(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True)
    fig.suptitle("Raw throughput: SSOG (solid) vs. dense (dashed)", fontsize=13,
                 fontweight="bold", x=0.06, ha="left", y=0.995)
    for ax, family in zip(axes.flat, FAMILIES):
        part = frame[frame.model_family == family]
        for size in SIZES:
            for mixer, linestyle in (("dense", "--"), ("ssog", "-")):
                line = part[(part.model_size == size) & (part.model_mixer == mixer)].sort_values("resolution")
                if line.empty:
                    continue
                ax.plot(line.resolution, line.images_per_second, marker="o", markersize=5,
                        color=COLORS[size], ls=linestyle, alpha=1.0 if mixer == "ssog" else 0.55)
            ssog_line = part[(part.model_size == size) & (part.model_mixer == "ssog")].sort_values("resolution")
            if not ssog_line.empty:
                last = ssog_line.iloc[-1]
                end_label(ax, last.resolution, last.images_per_second, size.title(), COLORS[size])
        ax.set_title(FAMILY_TITLES[family], fontsize=11, fontweight="bold", color=INK_PRIMARY)
        ax.set_xlabel("Input resolution")
        ax.set_ylabel("Throughput (images/s, log)")
        ax.set_yscale("log")
        ax.set_xticks(X_TICKS)
        ax.margins(x=0.16)
        ax.grid(True, which="major", axis="y", alpha=0.6)
        ax.grid(False, axis="x")
    size_handles = [plt.Line2D([], [], color=COLORS[s], marker="o", label=s.title()) for s in SIZES]
    mixer_handles = [plt.Line2D([], [], color=INK_SECONDARY, ls="--", alpha=0.55, label="Dense"),
                      plt.Line2D([], [], color=INK_SECONDARY, ls="-", label="SSOG")]
    fig.legend(handles=size_handles + mixer_handles, loc="upper center", ncol=6,
               frameon=False, bbox_to_anchor=(0.56, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, output, "paired_throughput_resolution")


SIZE_SWATCH = "#75736c"  # mid-gray, hollow, for the shape legend — distinct from
                          # both the hollow-family-colored Dense marks and the
                          # filled-family-colored SSOG marks, so no two legend
                          # glyphs render identically.


def _blank_corner(part, x_col, y_col):
    """Pick whichever scatter corner (top-right / bottom-left) has fewer nearby
    points, in log space, so callouts land on genuinely empty plot area rather
    than a fixed guess."""
    if part.empty:
        return "top-right"
    x, y = part[x_col].to_numpy(), part[y_col].to_numpy()
    xn = (pd.Series(x).rank(pct=True)).to_numpy()
    yn = (pd.Series(y).rank(pct=True)).to_numpy()
    top_right = ((xn > 0.6) & (yn > 0.6)).sum()
    bottom_left = ((xn < 0.4) & (yn < 0.4)).sum()
    return "top-right" if top_right <= bottom_left else "bottom-left"


def pareto_snapshot(frame, output, resolutions=(224, 1024)):
    """Cost (memory) vs. performance (throughput), Dense->SSOG as connected pairs.

    One axis is cost, the other is performance, so a single point-pair shows
    the whole trade rather than splitting speedup and memory into separate
    ratio charts. Two panels (one per resolution regime), color = family,
    marker shape = size, fill = mixer (hollow Dense -> filled SSOG).
    """
    fig, axes = plt.subplots(1, len(resolutions), figsize=(15.0, 7.6))
    best = {"gain": 0.0}
    for ax, res in zip(axes, resolutions):
        part = frame[frame.resolution == res]
        for family in FAMILIES:
            color = FAMILY_COLORS[family]
            for size in SIZES:
                pair = part[(part.model_family == family) & (part.model_size == size)]
                dense = pair[pair.model_mixer == "dense"]
                ssog = pair[pair.model_mixer == "ssog"]
                if dense.empty or ssog.empty:
                    continue
                d, s = dense.iloc[0], ssog.iloc[0]
                marker = SIZE_MARKERS[size]
                ax.plot([d.peak_allocated_mib, s.peak_allocated_mib],
                         [d.images_per_second, s.images_per_second],
                         color=color, lw=1.4, alpha=0.55, zorder=1)
                ax.scatter(d.peak_allocated_mib, d.images_per_second, marker=marker, s=110,
                           facecolor="none", edgecolor=color, linewidth=1.8, zorder=2)
                ax.scatter(s.peak_allocated_mib, s.images_per_second, marker=marker, s=110,
                           facecolor=color, edgecolor="#fcfcfb", linewidth=1.0, zorder=3)
                gain = (s.images_per_second / d.images_per_second) * (d.peak_allocated_mib / s.peak_allocated_mib)
                if gain > best["gain"]:
                    best = dict(gain=gain, family=family, size=size, res=res,
                                tp_ratio=s.images_per_second / d.images_per_second,
                                mem_ratio=s.peak_allocated_mib / d.peak_allocated_mib)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.55)
        ax.grid(True, which="minor", alpha=0.25)
        ax.set_xlabel(f"Peak memory, MiB (log) — {res}px input, batch 8", fontsize=10.5)
        ax.tick_params(labelsize=9.5)

        corner = _blank_corner(part, "peak_allocated_mib", "images_per_second")
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        if corner == "top-right":
            note_x, note_y, ha, va = 0.97, 0.95, "right", "top"
        else:
            note_x, note_y, ha, va = 0.04, 0.05, "left", "bottom"
        ax.text(note_x, note_y, "○ Dense\n● SSOG (ours)", transform=ax.transAxes,
                fontsize=9, color=INK_SECONDARY, ha=ha, va=va, linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcfcfb", edgecolor=GRIDLINE))
    axes[0].set_ylabel("Throughput, images/s (log) — higher is better", fontsize=10.5)

    if best["gain"] > 0:
        label = (f"Best trade: {FAMILY_TITLES[best['family']]}-{best['size'].title()} @ {best['res']}px\n"
                  f"{best['tp_ratio']:.2g}× faster, {(1 - best['mem_ratio']) * 100:.0f}% less memory")
        target_ax = axes[resolutions.index(best["res"])]
        target_part = frame[frame.resolution == best["res"]]
        corner = _blank_corner(target_part, "peak_allocated_mib", "images_per_second")
        # opposite corner from the mixer note so the two callouts never collide
        if corner == "top-right":
            tx, ty, ha, va = 0.04, 0.95, "left", "top"
        else:
            tx, ty, ha, va = 0.97, 0.05, "right", "bottom"
        target_ax.text(tx, ty, label, transform=target_ax.transAxes, fontsize=9.5,
                        color=INK_PRIMARY, fontweight="bold", ha=ha, va=va, linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fcfcfb", edgecolor=FAMILY_COLORS[best["family"]],
                                  linewidth=1.4))

    family_handles = [Line2D([], [], color=FAMILY_COLORS[f], marker="o", ls="", markersize=9, label=FAMILY_TITLES[f])
                       for f in FAMILIES]
    size_handles = [Line2D([], [], color=SIZE_SWATCH, marker=SIZE_MARKERS[s], ls="", markerfacecolor="none",
                            markeredgewidth=1.6, markersize=9, label=s.title()) for s in SIZES]
    fig.legend(handles=family_handles + size_handles, loc="upper center", ncol=8,
               frameon=False, bbox_to_anchor=(0.53, 1.0), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, output, "efficiency_pareto_snapshot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-family-detail", action="store_true",
                         help="Also emit the per-family speedup/memory ratio grids (superseded by the "
                              "combined Pareto snapshot, kept for appendix use).")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    frame = aggregate(args.input)
    frame = frame[frame.batch_size == args.batch_size]

    pareto_snapshot(frame, output)
    throughput_resolution(frame, output)

    if args.per_family_detail:
        speedup = paired_metric(frame, "images_per_second", lambda s, d: s / d)
        panels(speedup, "SSOG throughput ÷ dense throughput", "Throughput ratio (higher = SSOG faster)",
               "paired_ssog_speedup", output, parity_label="dashed line = parity")
        memory = paired_metric(frame, "peak_allocated_mib", lambda s, d: s / d)
        panels(memory, "SSOG peak VRAM ÷ dense peak VRAM", "Memory ratio (lower = SSOG lighter)",
               "paired_ssog_memory_ratio", output, parity_label="dashed line = parity")


if __name__ == "__main__":
    main()
