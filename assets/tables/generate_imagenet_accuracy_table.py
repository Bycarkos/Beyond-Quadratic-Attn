"""ImageNet-1k top-1/3/5 accuracy at dim768: SSOG vs dense (MHA), one row
pair per architecture family. Bold marks the best result in each column
within each family. Blank cells mark configurations with no data."""

import json
from pathlib import Path

DATA = json.load(open("experiments/SSOG/imagenet_results_2026-09-21.json"))
DIM = 768

FAMILY_ORDER = ["continuous", "discrete", "looped"]
FAMILY_LABELS = {
    "continuous": ("ODEViT (dense)", "SSOG (continuous)"),
    "discrete": ("ViT-B/16 (dense)", "SSOG (discrete)"),
    "looped": ("bViT (dense)", "SSOG (looped)"),
}
FAMILY_ROWCOLOR = {"continuous": "selectedrow", "discrete": "tertiaryrow", "looped": "secondaryrow"}

# Prefer the flagship (fixed, r4) continuous SSOG run at dim768 over the r2/steered variants.
PREFERRED_KEY_AT_DIM768 = {"continuous_ssog_dim768_r4"}

METRICS = ["accuracy_top_1", "accuracy_top_3", "accuracy_top_5"]


def index_runs():
    idx = {}
    for r in DATA["runs"]:
        fam = r["architecture_family"]
        mixer = r["attention_operator"]
        dim = r["model_hyperparameters"]["dimension"]
        key = (fam, mixer, dim)
        if key in idx and r["key"] not in PREFERRED_KEY_AT_DIM768:
            continue
        idx[key] = r
    return idx


def main():
    idx = index_runs()

    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"  \caption{ImageNet-1k accuracy at dim768: SSOG vs.\ dense")
    out.append(r"  multi-head attention (MHA), continuous/discrete/looped. Row color")
    out.append(r"  denotes architecture family (orange: continuous, blue: discrete,")
    out.append(r"  purple: looped); bold marks the best result in each column between")
    out.append(r"  the dense and SSOG variant within each family. `--' marks a")
    out.append(r"  configuration with no data (never launched or cancelled before a")
    out.append(r"  usable checkpoint); `*' marks a run still in progress, reporting")
    out.append(r"  its best validation accuracy so far rather than a final number.}")
    out.append(r"  \label{tab:ssog-imagenet-accuracy}")
    out.append(r"  \centering")
    out.append(r"  \begin{tabular}{lccc}")
    out.append(r"    \toprule")
    out.append(r"    Model & Acc@1 & Acc@3 & Acc@5 \\")
    out.append(r"    \midrule")

    for fam in FAMILY_ORDER:
        dense_label, ssog_label = FAMILY_LABELS[fam]
        dense_r = idx.get((fam, "dense", DIM))
        ssog_r = idx.get((fam, "ssog", DIM))

        best = {}
        for metric in METRICS:
            vals = []
            if dense_r is not None:
                vals.append(dense_r["validation_metrics"][metric])
            if ssog_r is not None:
                vals.append(ssog_r["validation_metrics"][metric])
            best[metric] = max(vals) if vals else None

        def row_cells(r):
            cells = []
            for metric in METRICS:
                if r is None:
                    cells.append("--")
                    continue
                val = r["validation_metrics"][metric] * 100
                text = f"{val:.2f}"
                if best[metric] is not None and abs(r["validation_metrics"][metric] - best[metric]) < 1e-9:
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            return cells

        def row_suffix(r):
            return "*" if r is not None and r["status"] != "completed" else ""

        out.append(f"    \\rowcolor{{{FAMILY_ROWCOLOR[fam]}}}")
        out.append(f"    {dense_label}{row_suffix(dense_r)} & " + " & ".join(row_cells(dense_r)) + r" \\")
        out.append(f"    \\rowcolor{{{FAMILY_ROWCOLOR[fam]}}}")
        out.append(f"    {ssog_label}{row_suffix(ssog_r)} & " + " & ".join(row_cells(ssog_r)) + r" \\")

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")

    text = "\n".join(out)
    print(text)
    Path("experiments/SSOG/imagenet_accuracy_table.tex").write_text(text + "\n")


if __name__ == "__main__":
    main()
