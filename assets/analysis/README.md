# assets/analysis

Post-hoc analyses of trained checkpoints. Both scripts are read-only: they load
models and validation images and write results, but never change a checkpoint
or config. Run from the repository root.

## analyze_attention_sinks.py

Measures whether trained models develop **attention sinks** (tokens that
receive far more than uniform attention from every query), comparing dense
multi-head attention (MHA) with SSoG.

For ten models (continuous, discrete and looped on CIFAR-100, fixed and
steered SSoG, and the discrete pair on ImageNet) it computes, per layer and
head, on validation images:

| Metric | Meaning |
|---|---|
| sink score | attention received by the most-attended key, relative to uniform (1 = no sink) |
| same key (`consistency`) | share of images whose most-attended key is the head's most frequent one |
| rollout sink | sink score of the residual attention rollout ∏(A + I)/2 over layers |
| CLS mass | attention from patch queries to the class token |
| entropy | mean query entropy / log N |
| high-norm | share of final patch tokens with norm > 3× the image median |

Dense attention maps are recomputed from each MHA layer's own weights; SSoG maps
come from the model's own dense-equivalent attention.

**Run**

```bash
sbatch scripts/evaluation/attention_sinks/analyze_attention_sinks.sh      # all models, then the LaTeX table
python -m assets.analysis.analyze_attention_sinks --num-images 512 --device cuda
python -m assets.analysis.analyze_attention_sinks --only "Discrete/imagenet/MHA"
```

| Option | Default | |
|---|---|---|
| `--num-images` | 512 | validation images per model |
| `--batch-size` | 32 | |
| `--device` | `cuda` if available | |
| `--only` | all | subset, as `Setting/dataset/Operator` |

**Needs** the checkpoints referenced by the evaluation configs
(`configs/classification/evaluation/patch_size/`), CIFAR-100 and the local
ImageNet validation shard. About one minute for all ten models on one A40;
much slower on CPU.

**Writes** `experiments/SSOG/attention_sinks/per_layer.csv` and `summary.csv`.
`--only` overwrites both files with just the selected models. Build the
table with `python -m assets.tables.generate_attention_sink_table`.

**Expected results** (512 images): the ImageNet ViT-B baseline shows a clear
sink (worst-layer score ≈ 40.6, rollout ≈ 41.4, same key in 91% of images);
fixed SSoG stays at or below ≈ 3.3 in every layer, with its most-attended key
always at an image corner. The script also prints each model's top-1 accuracy
on the analysed images, as a check that the checkpoint loaded correctly.

## analyze_ssog_sigma_anisotropy.py

Splits the Gaussian widths of SSoG into their two axes (σ_y, σ_x) and reports
the anisotropy ratio max(σ_y, σ_x) / min(σ_y, σ_x) per head and atom. For fixed
SSoG, σ comes straight from the checkpoint; for steered SSoG it is evaluated on
the given images and query patch.

**Run**

```bash
python -m assets.analysis.analyze_ssog_sigma_anisotropy \
  --checkpoints fixed:ssog/cifar100/fixed_r4_nfe12.yaml:<fixed.ckpt> \
                steered:ssog/cifar100/steered_r4_nfe12.yaml:<steered.ckpt> \
  --indices 0 1 2 3 --query 7 7
```

| Option | Default | |
|---|---|---|
| `--checkpoints` | required | one or more `label:model_config:checkpoint` entries (config under `configs/classification`) |
| `--indices` | `0 ...` | validation images used for steered models |
| `--query` | `7 7` | query patch (row, column) for steered models |
| `--output` | `experiments/SSOG/sigma_anisotropy/summary.json` | |

**Needs** the listed checkpoints; a GPU is used when available.

**Writes** a JSON summary with per-axis σ statistics and anisotropy ratios.
