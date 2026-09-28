"""Attention-sink table (MHA vs SSoG) from assets/analysis/analyze_attention_sinks.py's
summary.csv. Read-only with respect to the rest of the code base."""

import csv
import math
from pathlib import Path

SUMMARY = Path("experiments/SSOG/attention_sinks/summary.csv")
OUTPUT = Path("experiments/SSOG/attention_sink_table.tex")
ORDER = [("cifar100", "Continuous"), ("cifar100", "Discrete"), ("cifar100", "Looped"), ("imagenet", "Discrete")]
OPERATORS = ["MHA", "SSoG", "SSoG steered"]
DATASET = {"cifar100": "CIFAR-100", "imagenet": "ImageNet"}


def num(value, digits=2, pct=False):
    v = float(value)
    if math.isnan(v):
        return "--"
    return f"{100 * v:.{digits}f}" if pct else f"{v:.{digits}f}"


def main():
    rows = {(r["dataset"], r["setting"], r["operator"]): r for r in csv.DictReader(SUMMARY.open())}
    out = [
        r"\begin{table}[t]",
        r"  \caption{Attention-sink diagnostics of trained models on 512 validation images. \emph{Sink score}: attention received by the most-attended key relative to uniform ($1$ = no sink), averaged over heads, as the mean over layers, for the worst layer, and for the residual attention rollout $\prod_l (A_l + I)/2$ of the head-averaged maps. \emph{Same key}: share of images whose most-attended key is the head's most frequent one (averaged over heads; most consistent layer); for fixed SSoG it is $100\%$ by construction, because its content-independent kernel always peaks at an image corner through border renormalization. \emph{CLS}: attention mass from patch queries to the class token (-- where the model has none; SSoG patches never attend to it). \emph{High-norm}: share of final pre-norm patch tokens with norm above $3\times$ the per-image median.}",
        r"  \label{tab:attention-sinks}",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    & & \multicolumn{3}{c}{Sink score} & & & \\",
        r"    \cmidrule(lr){3-5}",
        r"    Setting & Operator & Mean & Worst layer & Rollout & Same key (\%) & CLS (\%) & High-norm (\%) \\",
    ]
    for dataset, setting in ORDER:
        present = [op for op in OPERATORS if (dataset, setting, op) in rows]
        if not present:
            continue
        out.append(r"    \midrule")
        for i, op in enumerate(present):
            r = rows[dataset, setting, op]
            head = f"\\multirow{{{len(present)}}}{{*}}{{{setting} ({DATASET[dataset]})}}" if i == 0 else ""
            out.append("    " + " & ".join([
                head, op, num(r["sink_mean"], 1), num(r["sink_max_layer"], 1), num(r["rollout_sink"], 1),
                num(r["consistency_max"], 0, pct=True), num(r["cls_mass"], 1, pct=True),
                num(r["high_norm_frac"], 1, pct=True),
            ]) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    text = "\n".join(out)
    OUTPUT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
