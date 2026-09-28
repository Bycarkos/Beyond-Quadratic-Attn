# assets

Everything that turns trained models and raw measurements into the numbers,
tables and figures of the SSoG paper. Nothing in here trains a model: training
lives in the `main_*.py` entry points at the repository root and their Slurm
launchers in `scripts/`.

```
assets/
  analysis/        post-hoc analyses of trained checkpoints (attention sinks, σ anisotropy)
  benchmarking/    throughput, memory, FLOP and profiling measurements
  plotting/        paper figures (PDF + PNG)
  tables/          paper tables (LaTeX)
  visualizations/  qualitative visualizations and the interactive supplementary page
```

Each folder has its own README listing every script, how to run it, what it
needs and what it produces.

## How to run anything in here

Run from the **repository root**, as a module:

```bash
cd /path/to/this/repository
python -m assets.plotting.plot_width_scaling
python -m assets.tables.generate_flops_table
```

The folders are Python packages, so the scripts can import each other and the
rest of the code (`models`, `eval_patch_size`, ...). Running a file directly
(`python assets/plotting/plot_width_scaling.py`) fails with an import error.

Use the environment the models were trained with (PyTorch, Lightning, Hydra,
matplotlib, scienceplots, torchdiffeq, datasets). Launchers in `scripts/`
select it automatically through `scripts/common.sh`.

## What needs what

| Folder | Hardware | Reads | Writes |
|---|---|---|---|
| `analysis/` | GPU recommended (CPU works, slowly) | checkpoints, validation images | `experiments/SSOG/attention_sinks/`, `experiments/SSOG/sigma_anisotropy/` |
| `benchmarking/` | GPU (except `count_flops.py`, CPU) | `configs/benchmarks/*.yaml` | `experiments/SSOG/efficiency/<run>/` |
| `plotting/` | CPU | result CSVs (and checkpoints for the two attention figures) | `experiments/SSOG/figures/{efficiency,attention}/` |
| `tables/` | CPU | result CSVs / JSON (one reads W&B) | `experiments/SSOG/*.tex` |
| `visualizations/` | CPU or GPU | checkpoints, results | `experiments/SSOG/supplementary/`, `artifacts/` |

Paths inside the scripts are relative to the repository root. Checkpoints are
located through the training configs (`modeling.checkpoint_path`) or through
the evaluation configs in `configs/classification/evaluation/`.

## Typical order

1. Train (`scripts/supervised/training/...`) and evaluate
   (`scripts/evaluation/patch_size/eval_patch_size.sh`).
2. Measure: `benchmarking/` (efficiency on a GPU, FLOPs on CPU).
3. Analyse: `analysis/` (attention sinks, anisotropy).
4. Produce the paper: `tables/` and `plotting/`.
5. Build the supplementary page: `visualizations/build_supplementary.py`.

## Slurm launchers that call these scripts

| Launcher | Runs |
|---|---|
| `scripts/supervised/benchmarking/benchmark_efficiency_resolution.sh` | `benchmarking.benchmark_efficiency_resolution` |
| `scripts/supervised/benchmarking/benchmark_efficiency_paired_a40.sh` | `benchmarking.benchmark_efficiency_resolution`, then `plotting.plot_efficiency_paired` |
| `scripts/supervised/benchmarking/benchmark_euler_rollout.sh` | `benchmarking.benchmark_euler_rollout` |
| `scripts/supervised/benchmarking/audit_ssog_apply_bottleneck.sh` | `benchmarking.audit_ssog_apply_bottleneck` |
| `scripts/evaluation/attention_sinks/analyze_attention_sinks.sh` | `analysis.analyze_attention_sinks`, then `tables.generate_attention_sink_table` |
| `scripts/supervised/visualization/plot_ssog_efficiency.sh` | `plotting.plot_efficiency_results` |
| `scripts/supervised/visualization/visualize_ssog_nfe.sh` | `visualizations.visualize_ssog_nfe` |
| `scripts/supervised/visualization/visualize_ssog_embeddings.sh` | `visualizations.visualize_ssog_embeddings` |

## A note on benchmark data

The efficiency numbers used in the paper come from the **isolated**
benchmarks: each model measured alone in its own process
(`configs/benchmarks/efficiency_*_isolated.yaml`). An earlier sweep that ran
several models in one process (`paired_base_a40`) was found to contaminate
memory and throughput readings. Scripts that still read older multi-model
sweeps are marked in the folder READMEs; treat their output as superseded.
