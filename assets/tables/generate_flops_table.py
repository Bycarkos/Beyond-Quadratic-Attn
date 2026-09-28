"""FLOPs table (per-image forward GFLOPs) for SSoG vs dense SDPA attention in
the three architecture settings, from assets/benchmarking/count_flops.py's CSV. For each setting:
whole-model and attention-module GFLOPs for both operators; bold marks the
lower count."""

import csv
from pathlib import Path

CSV = Path("experiments/SSOG/efficiency/flops/flops.csv")
OUTPUT = Path("experiments/SSOG/flops_table.tex")
FAMILIES = (("odevit", "Continuous"), ("vit", "Discrete"), ("bvit", "Looped"))


def fmt(value, other):
    text = f"{value:.1f}"
    return f"\\textbf{{{text}}}" if value < other else text


def main():
    rows = {}
    for r in csv.DictReader(CSV.open()):
        rows[r["family"], r["mixer"], int(r["resolution"])] = (float(r["total_gflops"]), float(r["attention_gflops"]))
    resolutions = sorted({k[2] for k in rows})

    out = [
        r"\begin{table*}[t]",
        r"  \caption{Forward FLOPs per image (GFLOPs, 2 per multiply-add) of the dim768 models used in the efficiency benchmarks, dense SDPA attention (MHA) versus SSoG, for the whole model and for the attention modules alone (including their projections). Element-wise operations (softmax, normalization, activations) are not counted for either operator. Bold marks the lower count.}",
        r"  \label{tab:flops}",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{c|cccc|cccc|cccc}",
        r"    \toprule",
        "     & " + " & ".join(f"\\multicolumn{{4}}{{c{'|' if i < 2 else ''}}}{{{title}}}" for i, (_, title) in enumerate(FAMILIES)) + r" \\",
        r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        "     & " + " & ".join(r"\multicolumn{2}{c}{Model} & \multicolumn{2}{c}{Attention}" for _ in FAMILIES) + r" \\",
        "    Resolution & " + " & ".join("MHA & SSoG & MHA & SSoG" for _ in FAMILIES) + r" \\",
        r"    \midrule",
    ]
    for res in resolutions:
        cells = [str(res)]
        for fam, _ in FAMILIES:
            (dt, da), (st, sa) = rows[fam, "dense", res], rows[fam, "ssog", res]
            cells += [fmt(dt, st), fmt(st, dt), fmt(da, sa), fmt(sa, da)]
        out.append("    " + " & ".join(cells) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    text = "\n".join(out)
    OUTPUT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
