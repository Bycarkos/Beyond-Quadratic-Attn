# assets/plotting

Paper figures. Every script writes a PDF (for LaTeX) and a PNG (for quick
viewing) and runs on CPU in seconds, except the two attention figures, which
load checkpoints and take about a minute each. Run from the repository root:

```bash
python -m assets.plotting.<script>
```

All figures share one style defined in `plot_ablation_horizontal.py`
(`configure_style`): serif type, black axes, a light horizontal grid,
Okabe–Ito colours (SSoG blue `#0072B2`, dense attention vermillion `#D55E00`,
looped setting green `#009E73`). Change the style there and every figure that
imports it follows.

Output folders: `experiments/SSOG/figures/efficiency/` and
`experiments/SSOG/figures/attention/`.

## Efficiency

| Script | Output | Data |
|---|---|---|
| `plot_efficiency_delta_by_resolution.py` | `efficiency_delta_by_resolution` | the six isolated benchmark CSVs + `efficiency/flops/flops.csv` |
| `plot_efficiency_dual_axis.py` | `efficiency_dual_axis_all_families_batch{8,32,...}` | the six isolated benchmark CSVs |
| `plot_efficiency_dual_axis_grid.py` | `efficiency_dual_axis_grid_all_batches` | same (via `plot_efficiency_dual_axis`) |
| `plot_efficiency_dual_axis_by_batch.py` | `efficiency_dual_axis_by_batch_res1024` | `paired_base_a40` CSVs (superseded, see below) |
| `plot_efficiency_results.py` | `efficiency_overview`, `efficiency_normalized_scaling`, `efficiency_pareto_512`, `efficiency_speedup`, `efficiency_*_all_settings`, `efficiency_model_sizes_{512,1024}` | `resolution_extended/`, `batch_scaling_512/`, `transformer_sizes/` |
| `plot_efficiency_paired.py` | `paired_throughput_resolution`, `efficiency_pareto_snapshot`, ... | any paired benchmark CSV (`--input`) |

- **`plot_efficiency_delta_by_resolution.py`**: improvement of SSoG over dense
  attention (%) per resolution, one panel per setting: measured throughput and
  memory saved, plus a fourth panel with FLOPs saved (identical for all
  settings). Above zero is better for SSoG.
- **`plot_efficiency_dual_axis.py`** takes `--csv`, `--output-dir` and
  `--batches`; **`..._by_batch.py`** takes `--csv`, `--output-dir` and
  `--resolutions`. Both draw throughput and memory on two axes.
- **`plot_efficiency_results.py`** takes `--extended`, `--batch`, `--sizes` and
  `--output-dir`. Its launcher is
  `scripts/supervised/visualization/plot_ssog_efficiency.sh`.
- **`plot_efficiency_paired.py`** requires `--input <resolution_scaling.csv>`
  and `--output-dir`, and is run by
  `scripts/supervised/benchmarking/benchmark_efficiency_paired_a40.sh`.

**Superseded data:** `plot_efficiency_dual_axis_by_batch.py` still reads the
contaminated `paired_base_a40` sweep, and `plot_efficiency_results.py` reads
older sweeps that measured several models in one process. Prefer the figures
built from the isolated benchmarks for any final claim.

## Accuracy and ablations

| Script | Output | What it shows |
|---|---|---|
| `plot_width_scaling.py` | `width_scaling` | CIFAR-100 Acc@1 vs parameters for D = 384/576/768, SSoG vs dense, per setting, with the D = 768 gain |
| `plot_ablation_horizontal.py` | `ablation_horizontal` | Acc@1 when heads, atoms or depth change, one line per setting |
| `plot_ablation_grid.py` | `ablation_grid` | the same ablations as a 3 × 3 grid with per-panel swings |
| `plot_accuracy_radar.py` | `accuracy_radar_families` | CIFAR-100 Acc@1 per setting, SSoG vs dense |
| `plot_accuracy_throughput_pareto.py` | `accuracy_throughput_pareto` | accuracy vs throughput, marker size = parameters |
| `plot_dense_vs_ssog_settings.py` | `dense_vs_ssog_settings` | accuracy bars per width and setting, with batch-1 latency |
| `plot_visual_abstract.py` | `visual_abstract` | architecture-scaling scatter next to the accuracy radar |

These scripts hold their accuracy numbers in a data block at the top of the
file (copied from the paper's tables). Update that block when a result changes.
`plot_visual_abstract.py` also reads `efficiency/transformer_sizes/`.

## Attention (qualitative)

Both scripts rebuild the models with `eval_patch_size.build_model`, so the
plotted kernels are exactly the evaluated ones. They need the checkpoints
referenced by `configs/classification/evaluation/patch_size/`, CIFAR-100 and
the local ImageNet validation shard. CPU is enough (about one minute each).

| Script | Output | What it shows |
|---|---|---|
| `plot_ssog_patch_size_attention.py` | `patch_size_attention_{cifar100_looped,imagenet_discrete}[_multiquery]` | one head per row at patch 16, 14, 8, and 8 with rescaled μ, σ; single and five-query versions |
| `plot_ssog_depth_attention.py` | `depth_attention_{imagenet_discrete,cifar100_looped}` | discrete kernels per block, and looped cumulative mixing per step (attention rollout) |

Image indices, the query pixel and the ImageNet layer are constants at the top
of each script.

## Checking a figure

Open the PNG next to the PDF. After changing data, rerun the script and the
paper picks up the new PDF on the next LaTeX build (the files are included by
name from `experiments/SSOG/figures/...`).
