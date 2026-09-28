# assets/tables

LaTeX tables for the paper, built from result files so every number in a table
can be traced to its source. They are not needed to reproduce the experiments
themselves. All run on CPU in about a second. Run from the repository root:

```bash
python -m assets.tables.<script>
```

Each script prints the table and writes it to `experiments/SSOG/<name>.tex`.
Paste it into `main.tex` or `\input{}` it.

| Script | Output | Reads | Produced by |
|---|---|---|---|
| `generate_flops_table.py` | `flops_table.tex` | `efficiency/flops/flops.csv` | `benchmarking/count_flops.py` |
| `generate_delta_by_resolution_table.py` | `delta_by_resolution_table.tex` | the six isolated benchmark CSVs | `benchmarking/benchmark_efficiency_resolution.py` |
| `generate_patch_size_table.py` | `patch_size_table.tex` | `patch_size_eval/cifar100/*.csv` | `eval_patch_size.py` |
| `generate_attention_sink_table.py` | `attention_sink_table.tex` | `attention_sinks/summary.csv` | `analysis/analyze_attention_sinks.py` |
| `generate_imagenet_accuracy_table.py` | `imagenet_accuracy_table.tex` | `imagenet_results_2026-09-21.json` | collected ImageNet results |
| `generate_capacity_ablation_tables.py` | `capacity_ablation_tables.tex` | Weights & Biases | training runs |

## What each table shows

- **FLOPs**: forward GFLOPs per image, whole model and attention modules, dense
  attention (MHA) vs SSoG, for each setting and resolution 224–1024 px. Bold
  marks the lower count.
- **Delta by resolution**: SSoG vs dense throughput and memory difference (%)
  per resolution, averaged over the seven batch sizes. Bold marks where SSoG
  wins.
- **Patch-size robustness**: CIFAR-100 Acc@1 at patch 16 (training), 14 and 8,
  for dense attention, SSoG, and SSoG with μ, σ rescaled to the new grid;
  change relative to patch 16 in parentheses.
- **Attention sinks**: sink score (mean, worst layer, rollout), same-key rate,
  CLS mass and high-norm share per model.
- **ImageNet accuracy**: top-1/3/5 at D = 768, SSoG vs dense, per setting.
- **Capacity ablations**: Heads / Atoms / Depth blocks for discrete and looped
  SSoG and their dense counterparts; unfinished runs show "--".

## Notes

- `generate_capacity_ablation_tables.py` queries **Weights & Biases** over the
  network with the project and entity set at the top of the file. It needs your
  W&B login and cannot run offline or for someone without access.
- `generate_imagenet_accuracy_table.py` reads a hand-collected JSON snapshot of
  the ImageNet runs; update the JSON when a run changes.
- Tables use `booktabs` and `multirow`; the capacity tables also use
  `\textcolor` (package `xcolor`).
