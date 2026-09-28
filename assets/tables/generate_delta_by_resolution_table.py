"""Delta-by-resolution table (bold-based win/loss, no color), matching the
user-provided style. Data comes exclusively from the six fully isolated
per-model benchmark re-runs (each model run alone in its own process),
which replace the original paired_base_a40 / paired_base_a40_batches sweep
after a cross-model memory/throughput contamination bug was found there."""

import csv
from pathlib import Path

ISOLATED_CSVS = {
    ("odevit", "dense"): Path("experiments/SSOG/efficiency/odevit_dense_isolated/resolution_scaling.csv"),
    ("odevit", "ssog"): Path("experiments/SSOG/efficiency/odevit_ssog_isolated/resolution_scaling.csv"),
    ("vit", "dense"): Path("experiments/SSOG/efficiency/vit_dense_isolated/resolution_scaling.csv"),
    ("vit", "ssog"): Path("experiments/SSOG/efficiency/vit_ssog_isolated/resolution_scaling.csv"),
    ("bvit", "dense"): Path("experiments/SSOG/efficiency/bvit_dense_isolated/resolution_scaling.csv"),
    ("bvit", "ssog"): Path("experiments/SSOG/efficiency/bvit_ssog_isolated/resolution_scaling.csv"),
}

RESOLUTIONS = [224, 384, 512, 640, 768, 1024]
BATCHES = [1, 8, 32, 64, 128, 256, 512]

FAMILY_ORDER = ["odevit", "vit", "bvit"]
FAMILY_LABELS = {"odevit": "Continuous", "vit": "Discrete", "bvit": "Looped"}


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


def delta(a, b):
    return (b - a) / a * 100


def thr_text(value):
    text = f"{value:+.1f}"
    return f"\\textbf{{{text}}}" if value > 0 else text


def mem_text(value):
    text = f"{value:+.1f}"
    return f"\\textbf{{{text}}}" if value < 0 else text


def main():
    throughput, memory = load()

    thr_by_res = {fam: {res: [] for res in RESOLUTIONS} for fam in FAMILY_ORDER}
    mem_by_res = {fam: {res: [] for res in RESOLUTIONS} for fam in FAMILY_ORDER}
    thr_all = {fam: [] for fam in FAMILY_ORDER}
    mem_all = {fam: [] for fam in FAMILY_ORDER}

    for res in RESOLUTIONS:
        for b in BATCHES:
            for fam in FAMILY_ORDER:
                dt = throughput.get((fam, "dense", res, b))
                st = throughput.get((fam, "ssog", res, b))
                dm = memory.get((fam, "dense", res, b))
                sm = memory.get((fam, "ssog", res, b))
                if dt is None or st is None or dm is None or sm is None:
                    continue
                thr_d = delta(dt, st)
                mem_d = delta(dm, sm)
                thr_by_res[fam][res].append(thr_d)
                mem_by_res[fam][res].append(mem_d)
                thr_all[fam].append(thr_d)
                mem_all[fam].append(mem_d)

    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"  \caption{SSOG vs.\ dense throughput and memory difference (\%) on an")
    out.append(r"  A40, collapsed by resolution (mean over all 7 measured batch sizes")
    out.append(r"  per resolution). Bold marks where SSOG wins (higher throughput or")
    out.append(r"  lower memory than dense); plain values mark where it loses.}")
    out.append(r"  \small")
    out.append(r"  \label{tab:ssog-throughput-delta-by-resolution}")
    out.append(r"  \vspace{3mm}")
    out.append(r"  \centering")
    out.append(r"  \setlength{\tabcolsep}{4pt}")
    out.append(r"  \begin{tabular}{ccccccc}")
    out.append(r"    \toprule")
    out.append(r"     & \multicolumn{2}{c}{Continuous} & \multicolumn{2}{c}{Discrete} & \multicolumn{2}{c}{Looped} \\")
    out.append(r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    out.append(r"    Resolution & Thr.& Mem. & Thr. & Mem.  & Thr. & Mem. \\")
    out.append(r"    \midrule")

    for res in RESOLUTIONS:
        row = [str(res)]
        for fam in FAMILY_ORDER:
            thr_vals = thr_by_res[fam][res]
            mem_vals = mem_by_res[fam][res]
            if not thr_vals:
                row.append("--")
                row.append("--")
                continue
            mean_thr = sum(thr_vals) / len(thr_vals)
            mean_mem = sum(mem_vals) / len(mem_vals)
            row.append(thr_text(mean_thr))
            row.append(mem_text(mean_mem))
        out.append("    " + " & ".join(row) + r" \\")

    out.append(r"    \midrule")
    mean_row = ["Mean"]
    for fam in FAMILY_ORDER:
        mean_thr = sum(thr_all[fam]) / len(thr_all[fam])
        mean_mem = sum(mem_all[fam]) / len(mem_all[fam])
        mean_row.append(thr_text(mean_thr))
        mean_row.append(mem_text(mean_mem))
    out.append("    " + " & ".join(mean_row) + r" \\")

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")

    text = "\n".join(out)
    print(text)
    Path("experiments/SSOG/delta_by_resolution_table.tex").write_text(text + "\n")


if __name__ == "__main__":
    main()
