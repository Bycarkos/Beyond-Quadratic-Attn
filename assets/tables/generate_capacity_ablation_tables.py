"""Capacity/depth ablation tables on CIFAR-100 (Heads, Atoms, Depth blocks)
for the discrete and looped SSoG models and their dense-MHA counterparts
(ViT, bViT), in the style of tab:ssog-capacity. Metrics are final-epoch
val/* summaries pulled from W&B; only finished runs are filled in, the rest
are shown as "--". Re-run as the pending runs finish."""

import os
from pathlib import Path

import wandb

# W&B project as "<entity>/<project>"; set SSOG_WANDB_PROJECT to your own.
PROJECT = os.environ.get("SSOG_WANDB_PROJECT", "<your-entity>/SSOG-Paper")
OUTPUT = Path("experiments/SSOG/capacity_ablation_tables.tex")
METRICS = ("acc_1", "acc_3", "acc_5", "loss")

# Discrete SSoG ran before the LayerNorm/GELU switch only for this run; the
# rest of its ablation uses the current architecture.
LEGACY_RUNS = {"cifar100-discrete-dim768-h12-r4-depth12-seed42"}

FAMILIES = {
    "discrete": dict(
        title="Discrete SSoG", depth_label="Depth", has_atoms=True,
        name="cifar100-discrete-dim768-h{h}-r{r}-depth{d}-seed42",
        params=lambda d: 0.83e6 + 5.9e6 * d,  # 36.23M at depth 6, 71.64M at depth 12
        label="tab:ssog-capacity-discrete",
    ),
    "looped": dict(
        title="Looped SSoG", depth_label="NFE", has_atoms=True,
        name="cifar100-looped-ssog-dim768-h{h}-r{r}-depth{d}-seed42",
        params=lambda d: 6.72e6,
        label="tab:ssog-capacity-looped",
    ),
    "vit": dict(
        title="Discrete MHA (ViT)", depth_label="Depth", has_atoms=False,
        name="cifar100-dense-vit-dim768-h{h}-depth{d}-seed42",
        params=lambda d: 0.83e6 + 7.0864e6 * d,  # 85.86M at depth 12
        label="tab:mha-capacity-discrete",
    ),
    "bvit": dict(
        title="Looped MHA (bViT)", depth_label="NFE", has_atoms=False,
        name="cifar100-bvit-dim768-h{h}-steps{d}-seed42",
        params=lambda d: 7.91e6,
        label="tab:mha-capacity-looped",
    ),
}

HEADS, ATOMS, DEPTHS = (6, 12, 24), (1, 2, 4, 8), (6, 8, 12, 16)


def fetch_runs():
    api = wandb.Api(timeout=60)
    runs = {}
    for run in api.runs(PROJECT, filters={"display_name": {"$regex": "^cifar100-(discrete|looped-ssog|dense-vit|bvit)-dim768-"}}):
        summary = run.summary
        runs[run.name] = dict(
            state=run.state, epoch=summary.get("epoch"),
            **{m: summary.get(f"val/{m}") for m in METRICS},
        )
    return runs


def block_rows(fam, block):
    """(varied value, heads, atoms, depth) for every row of one block."""
    if block == "Heads":
        return [(h, h, 4, 12) for h in HEADS]
    if block == "Atoms":
        return [(r, 12, r, 12) for r in ATOMS]
    return [(d, 12, 4, d) for d in DEPTHS]


def lookup(runs, fam, h, r, d):
    name = FAMILIES[fam]["name"].format(h=h, r=r, d=d)
    run = runs.get(name)
    if run is None:
        return name, None, "missing", None
    return name, run if run["state"] == "finished" else None, run["state"], run["epoch"]


def fmt(value, metric, best):
    text = f"{value:.4f}" if metric == "loss" else f"{100 * value:.2f}"
    return f"\\textbf{{{text}}}" if value == best else text


def table(fam, runs, status):
    spec = FAMILIES[fam]
    blocks = ["Heads", "Atoms", "Depth"] if spec["has_atoms"] else ["Heads", "Depth"]
    atom_col = " & $R$" if spec["has_atoms"] else ""
    cols = "c|ccc" + ("c" if spec["has_atoms"] else "") + "|ccccc"
    out = [
        r"\begin{table}[t]",
        f"  \\caption{{{spec['title']} capacity and depth ablations on CIFAR-100 "
        f"(final-epoch validation metrics). The column highlighted in blue is the "
        f"parameter varied within each block; bold marks the best metric within "
        f"each block; -- marks runs still training.}}",
        f"  \\label{{{spec['label']}}}",
        r"  \centering", r"  \small", r"  \vspace{5mm}",
        f"  \\begin{{tabular}}{{{cols}}}", r"    \toprule",
        f"    Ablation & $Dim$ & $Heads${atom_col} & {spec['depth_label']} & Params & Acc@1 & Acc@3 & Acc@5 & Loss \\\\",
    ]
    for block in blocks:
        rows = block_rows(fam, block)
        found = []
        for varied, h, r, d in rows:
            name, run, state, epoch = lookup(runs, fam, h, r, d)
            status.append((spec["title"], block, name, state, epoch))
            found.append((varied, h, r, d, name, run))
        best = {
            m: (min if m == "loss" else max)(run[m] for *_, run in found if run)
            for m in METRICS if any(run for *_, run in found)
        }
        out.append(r"    \midrule")
        for i, (varied, h, r, d, name, run) in enumerate(found):
            first = i == 0
            n = len(found)
            blue = f"\\textcolor{{blue}}{{{varied}}}"
            cell = lambda val, key: (blue if block == key else (f"\\multirow{{{n}}}{{*}}{{{val}}}" if first else ""))
            head = f"\\multirow{{{n}}}{{*}}{{\\textcolor{{blue}}{{{block}}}}}" if first else ""
            params = f"{spec['params'](d) / 1e6:.1f}M"
            if block != "Depth" or len({spec['params'](x) for x in DEPTHS}) == 1:
                params = f"\\multirow{{{n}}}{{*}}{{{params}}}" if first else ""
            cells = [head, r"\multirow{%d}{*}{768}" % n if first else "", cell(h, "Heads")]
            if spec["has_atoms"]:
                cells.append(cell(r, "Atoms"))
            cells += [cell(d, "Depth"), params]
            if run:
                values = [fmt(run[m], m, best.get(m)) for m in METRICS]
                if name in LEGACY_RUNS:
                    values[0] += r"$^\dagger$"
            else:
                values = ["--"] * 4
            out.append("    " + " & ".join(cells + values) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}"]
    if fam == "discrete":
        out.append(r"  \par\vspace{1mm}{\footnotesize $^\dagger$Trained with the earlier CenterNorm/ELU block; the other rows use the current LayerNorm/GELU block.}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    runs = fetch_runs()
    status = []
    tables = [table(fam, runs, status) for fam in FAMILIES]
    OUTPUT.write_text("\n\n".join(tables) + "\n")
    print("\n\n".join(tables))
    print(f"\nWrote {OUTPUT}\n")
    for title, block, name, state, epoch in status:
        print(f"{title:20s} {block:6s} {state:9s} epoch={epoch!s:5s} {name}")


if __name__ == "__main__":
    main()
