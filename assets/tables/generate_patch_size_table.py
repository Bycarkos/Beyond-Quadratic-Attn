"""Patch-size robustness table: SSoG vs dense MHA on CIFAR-100, evaluated at
patch sizes 16 (training), 14 and 8 with the image kept at 224px. Reads the
CSVs written by eval_patch_size.py. Delta is Acc@1 change vs patch 16."""

import csv
from pathlib import Path

RESULTS = Path("experiments/SSOG/patch_size_eval/cifar100")
OUTPUT = Path("experiments/SSOG/patch_size_table.tex")
PATCHES = (16, 14, 8)
FAMILIES = (("Continuous", "continuous"), ("Discrete", "discrete"), ("Looped", "looped"))
# (row label, csv prefix, geometry mode)
OPERATORS = (
    ("MHA", "dense_", "token"),
    ("SSoG", "", "token"),
    ("SSoG (rescaled $\\mu,\\sigma$)", "", "image"),
)


def load(name):
    with open(RESULTS / f"{name}.csv") as handle:
        return {(int(r["patch_size"]), r["geometry"]): float(r["acc_1"]) for r in csv.DictReader(handle)}


def acc(rows, patch, geometry):
    # At the training patch size both geometry modes are identical.
    return rows.get((patch, geometry), rows.get((patch, "token")))


def cell(value, delta, best):
    text = f"\\textbf{{{value:.2f}}}" if best else f"{value:.2f}"
    return text if delta is None else f"{text} {{\\scriptsize({delta:+.2f})}}"


def main():
    out = [
        r"\begin{table}[t]",
        r"  \caption{Patch-size robustness on CIFAR-100: models trained at patch 16 (224px, $14\times14$ tokens) evaluated at patch 14 and 8 without retraining. Patch-embedding kernels are PI-resized and position embeddings bicubically interpolated for every model; the rescaled SSoG row also scales $\mu$, $\sigma$ by the grid ratio so each Gaussian covers the same image area. Acc@1 (\%), change vs.\ patch 16 in parentheses; bold marks the best within each setting.}",
        r"  \label{tab:patch-size-robustness}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    & & Patch 16 & Patch 14 & Patch 8 \\",
        r"    Setting & Operator & ($14\times14$) & ($16\times16$) & ($28\times28$) \\",
    ]
    for title, fam in FAMILIES:
        grid = []
        for label, prefix, geometry in OPERATORS:
            rows = load(prefix + fam)
            grid.append([acc(rows, p, geometry) for p in PATCHES])
        best = [max(row[i] for row in grid) for i in range(len(PATCHES))]
        out.append(r"    \midrule")
        for j, ((label, _, _), row) in enumerate(zip(OPERATORS, grid)):
            cells = [
                cell(v, None if p == 16 else v - row[0], v == best[i])
                for i, (p, v) in enumerate(zip(PATCHES, row))
            ]
            if j == 0:
                head = f"\\multirow{{{len(OPERATORS)}}}{{*}}{{{title}}}"
            else:
                head = ""
            out.append("    " + " & ".join([head, label] + cells) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    text = "\n".join(out)
    OUTPUT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
