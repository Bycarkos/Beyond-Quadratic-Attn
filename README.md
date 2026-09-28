# SSoG: Separable Sum-of-Gaussians attention

Code for SSoG, a vision-transformer token mixer that replaces dense
query–key attention with a small, learned **mixture of separable Gaussian
kernels over token offsets**. Each head has R Gaussian atoms with a centre μ,
an anisotropic width σ and a weight λ; the kernel is applied as two 1-D
filtering passes (rows, then columns), so no N × N attention matrix is formed.
With **content steering**, each patch predicts bounded shifts of μ, rescalings
of σ and reweightings of λ from its own features.

SSoG is evaluated in three architecture settings, each against a matched
dense multi-head-attention baseline:

| Setting | SSoG model | Dense baseline |
|---|---|---|
| Continuous depth (neural ODE) | `SSOG_ViT` (`models/SSOGViT.py`) | ODE-ViT (`models/ODEViT.py`) |
| Discrete (12 independent blocks) | `DiscreteSSOG_ViT` (`models/DiscreteSSOGViT.py`) | ViT (`models/MatchedViT.py`) |
| Looped (one block applied 12 times) | `DiscreteSSOG_ViT` with `recurrent: true` | bViT (`models/MatchedViT.py`) |

## Repository layout

```
main_classification_*.py   training entry points (one per model family)
eval_patch_size.py         evaluation at unseen patch sizes
models/ trainers/ dataset/ model, Lightning trainers, data collation
configs/classification/    Hydra configs for every experiment (see below)
configs/benchmarks/        efficiency benchmark configs
scripts/                   Slurm launchers (scripts/common.sh is shared setup)
assets/                    benchmarks, analyses, figures, tables, supplementary page
experiments/SSOG/          result files behind the paper's figures and tables
```

`Reproducibility.md` lists every command behind the paper in order.
`assets/README.md` and the README in each `assets/` subfolder document every
script there.

## Installation

```bash
conda create -n ssog python=3.12 && conda activate ssog
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132   # match your CUDA
pip install -r requirements.txt
```

## Data

- **CIFAR-100** is read with `torchvision` from `~/cifar` (`data.dataset.dataset_path`).
  Configs set `download: false`, so download it once:
  `python -c "from torchvision.datasets import CIFAR100; CIFAR100('~/cifar', download=True)"`
- **ImageNet-1k** is read with Hugging Face `datasets` (`ILSVRC/imagenet-1k`,
  cache in `~/imagenet-1k`). Accept the dataset terms on the Hugging Face Hub
  and log in (`huggingface-cli login`) before the first run.

Images are resized to 224 × 224 and split into 16 × 16 patches (14 × 14 tokens).
Checkpoints are written to `~/checkpoints/<project>/<group>/<run name>`.

## Training

Every experiment is one config file. Pass it to the entry point of its model
family:

| Model family | Entry point | Configs |
|---|---|---|
| Continuous SSoG | `main_classification_ssog.py` | `ssog/cifar100/`, `ssog/imagenet/`, `ssog/augmentation/`, `ssog/cifar100_transfer/*.yaml` |
| Discrete and looped SSoG | `main_classification_discrete_ssog.py` | `ssog/discrete/`, `ssog/looped_ssog/`, `ssog/cifar100_transfer/{discrete,looped}/` |
| Dense ODE-ViT | `main_classification_odevit_baseline.py` | `ssog/baselines/*/dense_odevit_*` |
| Dense ViT (CIFAR-100) | `main_classification_dense_vit_baseline.py` | `ssog/baselines/cifar100/dense_vit_*` |
| Dense ViT-B/16 (ImageNet) | `main_classification_vit_baseline.py` | `ssog/baselines/imagenet/vit_b16*` |
| Dense bViT | `main_classification_bvit_baseline.py` | `ssog/baselines/*/bvit_*` |

Config paths are relative to `configs/classification/`.

**Without Slurm**

```bash
python main_classification_ssog.py -c ssog/cifar100/fixed_r4_nfe12.yaml
python main_classification_discrete_ssog.py -c ssog/looped_ssog/cifar100/fixed_dim768_h12_r4_depth12.yaml
```

**With Slurm**, submit from the repository root:

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_nfe12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/discrete/imagenet/fixed_dim768_h12_r4_depth12.yaml
```

The partitions and GPU types in the `#SBATCH` headers are examples; set your own
there or on the command line (`sbatch -p <partition> --gres=gpu:4 --ntasks-per-node=4 ...`).
Before training, each launcher checks that the config's
`setup.hardware.devices` matches the GPUs Slurm allocated, and stops with the
exact flags to use if it does not.

**Key config fields** (`modeling.inputs`): `dim`, `num_heads`, `num_atoms` (R),
`lookat` (content steering on/off), `max_offset` (steering bound for μ),
`sigma_floor`, and `num_eval_steps` (continuous: ODE steps + 1) or `depth`
(discrete and looped).

**Logging:** configs log to Weights & Biases (`log_wandb: true`). Run
`wandb login`, or set `WANDB_MODE=offline` (or `disabled`) to train without an
account.

## Evaluation and analyses

```bash
# accuracy at patch sizes 16, 14 and 8, with the learned geometry in token units and rescaled
sbatch scripts/evaluation/patch_size/eval_patch_size.sh evaluation/patch_size/cifar100/looped.yaml

# attention-sink diagnostics of all trained models, then the LaTeX table
sbatch scripts/evaluation/attention_sinks/analyze_attention_sinks.sh
```

Evaluation configs (`configs/classification/evaluation/`) point at the trained
checkpoints through the training configs they inherit from.

## Efficiency

```bash
# throughput and peak memory, one model per job (inference, fp16)
sbatch scripts/supervised/benchmarking/benchmark_efficiency_resolution.sh configs/benchmarks/efficiency_vit_ssog_isolated.yaml

# forward FLOPs of all models (CPU)
python -m assets.benchmarking.count_flops
```

Benchmark each model in its own process (the `*_isolated.yaml` configs):
measuring several models in one process contaminates memory and throughput
readings.

## Figures, tables and supplementary page

The result files in `experiments/SSOG/` are included, so the paper's figures
and tables can be rebuilt without a GPU. Run from the repository root:

```bash
python -m assets.tables.generate_flops_table
python -m assets.plotting.plot_efficiency_delta_by_resolution
python -m assets.plotting.plot_width_scaling
```

Figures go to `experiments/SSOG/figures/`, tables to `experiments/SSOG/*.tex`.
The two qualitative attention figures and the interactive supplementary page
(`python -m assets.visualizations.build_supplementary`) also need the trained
checkpoints.

| Result files | Produced by |
|---|---|
| `experiments/SSOG/efficiency/*_isolated/`, `flops/` | `assets/benchmarking/` |
| `experiments/SSOG/efficiency/{resolution_extended,batch_scaling_512,transformer_sizes,paired_*}/` | earlier benchmark sweeps (superseded by the isolated runs) |
| `experiments/SSOG/patch_size_eval/cifar100/` | `eval_patch_size.py` |
| `experiments/SSOG/attention_sinks/` | `assets/analysis/analyze_attention_sinks.py` |
| `experiments/SSOG/imagenet_results_2026-09-21.json` | collected ImageNet runs |

## License

Apache License 2.0, see `LICENSE`.
