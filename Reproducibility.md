# Reproducing the paper

This file lists, in order, every command behind the paper's experiments:
training (CIFAR-100, ImageNet-1k, transfer), evaluation, analyses, efficiency
benchmarks, and the figures and tables. See `README.md` for installation and
the repository layout.

All commands run from the repository root. Config paths are relative to
`configs/classification/`.

## 0. Setup

```bash
conda create -n ssog python=3.12 && conda activate ssog
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132   # match your CUDA
pip install -r requirements.txt

# CIFAR-100 (configs read it from ~/cifar with download: false)
python -c "from torchvision.datasets import CIFAR100; CIFAR100('~/cifar', download=True)"

# ImageNet-1k (Hugging Face ILSVRC/imagenet-1k, cached in ~/imagenet-1k): accept the terms on the Hub, then
huggingface-cli login

# Weights & Biases: log in, or run without an account
wandb login            # or: export WANDB_MODE=offline
```

**How runs are set up**

- **One config per experiment.** Each YAML composes shared Hydra blocks (data,
  augmentation, optimizer, model) through `defaults:` and overrides only what the
  experiment changes.
- **Launchers.** Every Slurm launcher sources `scripts/common.sh` and takes the
  config as its first argument. Before training it checks that the config's
  `setup.hardware.devices` matches the GPUs Slurm allocated. Partitions and GPU
  types in the `#SBATCH` headers are examples: override them on the command line
  (`sbatch -p <partition> ...`).
- **Without Slurm**, call the launcher's entry point directly, for example
  `python main_classification_ssog.py -c ssog/cifar100/fixed_r4_nfe12.yaml`.
  The entry point for each config family is listed in `README.md`.
- **Determinism.** Lightning runs with `seed_everything(seed)` and
  `deterministic=True`. The seed is set in `setup.experiment.seed`: 42 unless the
  config name says otherwise.
- **Outputs.** Checkpoints go to `~/checkpoints/<project>/<group>/<run name>`.
  Later steps (transfer, evaluation, analyses, qualitative figures) read
  checkpoints from those paths. If you train elsewhere, edit the
  `checkpoint` / `pretrained_checkpoint_path` fields they name.
- **Resumed runs.** A few configs continue an interrupted run of the paper and
  do not start from scratch as shipped. Section 9 says what to change in each.

## 1. CIFAR-100 (training from scratch)

Every run uses the recipe in `configs/classification/ssog/base_cifar100.yaml`
(and its discrete, looped and baseline counterparts): 224 × 224 input,
16 × 16 patches and strong augmentation. Unless noted, the SSoG default is
D = 768, 12 heads, R = 4 atoms and σ floor 0.25.

### 1.1 Continuous-depth SSoG (ODE)

Entry point `main_classification_ssog.py`; 1 GPU each; 500 epochs.

**Flagship (D = 768, 12 heads, R = 4, 12 ODE steps) and its two extra seeds**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h12_r4_nfe12_seed43.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h12_r4_nfe12_seed44.yaml
```

**Width (head dimension 64)**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim192_h3_r4_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim192_h3_r4_nfe12_seed43.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim384_h6_r4_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim576_h9_r4_nfe12.yaml
```

**Heads at D = 768**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h6_r4_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h24_r4_nfe12.yaml
```
The 12-head point is the flagship run.

**Atoms per head R**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r1_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r2_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r8_nfe12.yaml
```
R = 4 is the flagship run.

**Heads × atoms combinations**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h24_r2_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h6_r8_nfe12.yaml
```

**ODE steps (NFE)**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_nfe6.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_nfe8.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_nfe16.yaml
```
NFE = 12 is the flagship run.

**σ floor**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_sigma010.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_sigma050.yaml
```
The flagship uses σ floor 0.25.

**LayerNorm instead of CenterNorm**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim384_h6_r4_nfe12_layernorm.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim576_h9_r4_nfe12_layernorm.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_dim768_h6_r4_nfe12_layernorm.yaml
```

**Content steering: NFE**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_r4_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_r4_nfe8.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_r4_nfe6.yaml
```

**Content steering: bound on the μ shift (`max_offset`, default 7)**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_r4_offset2.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_r4_offset4.yaml
```

**Content steering: initialization of the steering gates**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_r4_warm_init.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_mu_only_init4.yaml
```

**Content steering: which components are steered**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_mu_only_cold.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_sigma_only_cold.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_lambda_only_cold.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/steered_mu_sigma_cold.yaml
```

**Augmentation strength**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/augmentation/cifar100/fixed_moderate.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/augmentation/cifar100/fixed_weak.yaml
```
Strong augmentation is the flagship recipe.


### 1.2 Discrete and looped SSoG

Entry point `main_classification_discrete_ssog.py` for both (looped configs set `recurrent: true`); 1 GPU each; 500 epochs, seed 42.

**Discrete: width, heads, atoms, depth, steering**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h12_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim384_h3_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim576_h9_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h6_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h24_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h12_r1_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h12_r2_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h12_r8_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h12_r4_depth6.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/fixed_dim768_h12_r4_depth8.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/discrete/cifar100/steered_dim768_h12_r4_depth12.yaml
```

**Looped: the same grid**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim384_h3_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim576_h9_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h6_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h24_r4_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r1_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r2_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r8_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r4_depth6.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r4_depth8.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/steered_dim768_h12_r4_depth12.yaml
```


### 1.3 Dense-attention baselines

Same data, recipe and widths as the SSoG models; only the token mixer differs. 1 GPU each.

**Continuous: dense ODE-ViT, width**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_dense_odevit_baseline.sh ssog/baselines/cifar100/dense_odevit_dim192_h3_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_odevit_baseline.sh ssog/baselines/cifar100/dense_odevit_dim384_h6_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_odevit_baseline.sh ssog/baselines/cifar100/dense_odevit_dim576_h9_nfe12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_odevit_baseline.sh ssog/baselines/cifar100/dense_odevit_dim768_h12_nfe12.yaml
```

**Discrete: ViT, width, heads, depth**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim192_h3_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim384_h6_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim576_h9_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim768_h12_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim768_h6_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim768_h24_depth12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim768_h12_depth6.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim768_h12_depth8.yaml
sbatch scripts/supervised/training/cifar100/cifar100_dense_vit_baseline.sh ssog/baselines/cifar100/dense_vit_dim768_h12_depth16.yaml
```

**Looped: bViT, width, heads, steps**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim384_h3_steps12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim576_h9_steps12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim768_h12_steps12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim768_h6_steps12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim768_h24_steps12.yaml
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim768_h12_steps6.yaml
sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim768_h12_steps8.yaml
```


## 2. ImageNet-1k

### 2.1 SSoG on ImageNet-1k

300 epochs. The commands carry the GPU flags each config needs (the device check in the launchers stops a mismatched job).

**Continuous: width, atoms, steering**

```bash
sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/fixed_dim384_h6_r4_nfe12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/fixed_dim576_h9_r4_nfe12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/fixed_r4_nfe12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/fixed_r2_nfe12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/steered_r4_nfe12.yaml
```
The D = 768 run (`fixed_r4_nfe12`) was interrupted and continued on one GPU with `fixed_r4_nfe12_resume.yaml` (see section 9):

```bash
sbatch --gres=gpu:1 --ntasks-per-node=1 --mem=40G scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/fixed_r4_nfe12_resume.yaml
```

**Discrete**

```bash
sbatch scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/discrete/imagenet/fixed_dim384_h6_r4_depth12.yaml
sbatch --gres=gpu:1 --ntasks-per-node=1 --mem=40G scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/discrete/imagenet/fixed_dim576_h9_r4_depth12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/discrete/imagenet/fixed_dim768_h12_r4_depth12.yaml
```

**Looped**

```bash
sbatch --gres=gpu:1 --ntasks-per-node=1 --mem=40G scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/looped_ssog/imagenet/fixed_dim384_h6_r4_depth12.yaml
sbatch --gres=gpu:1 --ntasks-per-node=1 --mem=40G scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/looped_ssog/imagenet/fixed_dim576_h9_r4_depth12.yaml
sbatch scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/looped_ssog/imagenet/fixed_dim768_h12_r4_depth12.yaml
```


### 2.2 Dense baselines on ImageNet-1k



**ViT-B/16, trained in three stages (run them in order)**

```bash
sbatch scripts/supervised/training/imagenet/imagenet_vit_baseline.sh ssog/baselines/imagenet/vit_b16.yaml
sbatch scripts/supervised/training/imagenet/imagenet_vit_baseline.sh ssog/baselines/imagenet/vit_b16_resume_e31_lr2e4.yaml
sbatch scripts/supervised/training/imagenet/imagenet_vit_baseline.sh ssog/baselines/imagenet/vit_b16_resume_raw_lr2e4.yaml
```
Stage 2 restarts from stage 1's epoch-31 checkpoint with base learning rate 2e-4; stage 3 continues stage 2 from its live (non-EMA) weights. Each stage's `resume_checkpoint_path` names the checkpoint of the previous one.

**Looped bViT-B/16**

```bash
sbatch scripts/supervised/training/imagenet/imagenet_bvit_baseline.sh ssog/baselines/imagenet/bvit_b16_steps12.yaml
```


## 3. ImageNet → CIFAR-100 transfer

Linear probe (frozen backbone, 50 epochs) and full fine-tuning (100 epochs), each with two augmentation strengths. **These need the ImageNet checkpoints of section 2.1**: set `pretrained_checkpoint_path` in the base config of each block to your checkpoint (the paper used the continuous D = 576 run at epoch 273, and the best discrete / looped D = 768 epochs, 293 and 299).

**Continuous (D = 576)**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100_transfer/imagenet_dim576_linear_probe.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100_transfer/imagenet_dim576_linear_weak_aug.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100_transfer/imagenet_dim576_full_finetune.yaml
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100_transfer/imagenet_dim576_full_moderate_aug.yaml
```

**Discrete (D = 768)**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/discrete/imagenet_dim768_linear_probe.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/discrete/imagenet_dim768_linear_weak_aug.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/discrete/imagenet_dim768_full_finetune.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/discrete/imagenet_dim768_full_moderate_aug.yaml
```

**Looped (D = 768)**

```bash
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/looped/imagenet_dim768_linear_probe.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/looped/imagenet_dim768_linear_weak_aug.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/looped/imagenet_dim768_full_finetune.yaml
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/cifar100_transfer/looped/imagenet_dim768_full_moderate_aug.yaml
```


## 4. Evaluation at unseen patch sizes

These runs evaluate trained models at patch sizes 16 (training), 14 and 8, keeping the image at 224 px. SSoG is evaluated with its geometry in token units and also with μ, σ rescaled to image units. Each config points at the checkpoint of the run it inherits from. Results go to `experiments/SSOG/patch_size_eval/`.

```bash
for m in continuous discrete looped dense_continuous dense_discrete dense_looped; do
  sbatch scripts/evaluation/patch_size/eval_patch_size.sh evaluation/patch_size/cifar100/$m.yaml
done
sbatch scripts/evaluation/patch_size/eval_patch_size.sh evaluation/patch_size/imagenet/discrete.yaml
```

## 5. Attention-sink analysis

This job computes the sink score, same-key consistency, rollout, CLS mass and high-norm tokens for every trained model on 512 validation images. It then writes the LaTeX table. Results go to `experiments/SSOG/attention_sinks/`.

```bash
sbatch scripts/evaluation/attention_sinks/analyze_attention_sinks.sh
# one model only, e.g.:
sbatch scripts/evaluation/attention_sinks/analyze_attention_sinks.sh --only "Discrete/imagenet/MHA"
```

Optional σ-anisotropy analysis of trained SSoG checkpoints (`assets/analysis/README.md`):

```bash
python -m assets.analysis.analyze_ssog_sigma_anisotropy --checkpoints <label>:<model config>:<checkpoint> [...]
```

## 6. Efficiency benchmarks

Throughput and peak memory use inference in fp16. Run each model in its own job: measuring several models in one process skews the readings. Results go to `experiments/SSOG/efficiency/<name>_isolated/`.

```bash
for f in odevit bvit vit; do
  for op in ssog dense; do
    sbatch scripts/supervised/benchmarking/benchmark_efficiency_resolution.sh configs/benchmarks/efficiency_${f}_${op}_isolated.yaml
  done
done

# attention module alone, SSoG vs SDPA
python -m assets.benchmarking.benchmark_attention_isolated

# forward FLOPs of all models (CPU; MHA fast path off, math SDPA backend)
python -m assets.benchmarking.count_flops
```

Earlier sweeps behind the appendix efficiency figures (superseded by the isolated runs above for the main results):

```bash
for c in resolution_extended batch_scaling_512 transformer_sizes paired_settings; do
  sbatch scripts/supervised/benchmarking/benchmark_efficiency_resolution.sh configs/benchmarks/efficiency_$c.yaml
done
sbatch scripts/supervised/benchmarking/benchmark_efficiency_paired_a40.sh
```

Diagnostics of the SSoG implementation (not in the paper's tables):

```bash
sbatch scripts/supervised/benchmarking/benchmark_euler_rollout.sh
sbatch scripts/supervised/benchmarking/audit_ssog_apply_bottleneck.sh --profile
python -m assets.benchmarking.benchmark_fixed_kernel_cache --batch-size 64
```

## 7. Tables and figures

The result files of sections 4–6 ship in `experiments/SSOG/`, so this step runs without a GPU. Tables go to `experiments/SSOG/*.tex` and figures to `experiments/SSOG/figures/`.

```bash
# tables
python -m assets.tables.generate_flops_table
python -m assets.tables.generate_delta_by_resolution_table
python -m assets.tables.generate_patch_size_table
python -m assets.tables.generate_attention_sink_table
python -m assets.tables.generate_imagenet_accuracy_table
SSOG_WANDB_PROJECT=<entity>/<project> python -m assets.tables.generate_capacity_ablation_tables   # reads the training runs from W&B

# efficiency figures
python -m assets.plotting.plot_efficiency_delta_by_resolution
python -m assets.plotting.plot_efficiency_dual_axis
python -m assets.plotting.plot_efficiency_dual_axis_grid
python -m assets.plotting.plot_efficiency_dual_axis_by_batch
python -m assets.plotting.plot_efficiency_results
python -m assets.plotting.plot_efficiency_paired --input <paired benchmark csv> --output-dir experiments/SSOG/figures/paired

# accuracy figures (numbers from the paper's tables)
python -m assets.plotting.plot_width_scaling
python -m assets.plotting.plot_ablation_horizontal
python -m assets.plotting.plot_ablation_grid
python -m assets.plotting.plot_accuracy_radar
python -m assets.plotting.plot_accuracy_throughput_pareto
python -m assets.plotting.plot_dense_vs_ssog_settings
python -m assets.plotting.plot_visual_abstract

# qualitative attention figures (need the trained checkpoints; a GPU helps)
python -m assets.plotting.plot_ssog_patch_size_attention
python -m assets.plotting.plot_ssog_depth_attention
```

## 8. Supplementary material

The interactive page is a single offline HTML file: `experiments/SSOG/supplementary/ssog_explorer.html`. It needs the trained checkpoints.

```bash
python -m assets.visualizations.build_supplementary
```

Other visualizations (NFE evolution, embeddings) take a model config and a checkpoint:

```bash
sbatch scripts/supervised/visualization/visualize_ssog_nfe.sh <model config> <checkpoint> [output dir]
sbatch scripts/supervised/visualization/visualize_ssog_embeddings.sh <model config> <checkpoint> [output dir]
```

## 9. Resumed runs and configs that do not start from scratch

These configs continue interrupted runs of the paper. As shipped, they expect that run's checkpoint or W&B id. To train one from scratch, make the change in the last column. The W&B ids point at the authors' project and will not exist in yours.

| Config | Why | For a fresh run |
|---|---|---|
| `ssog/baselines/cifar100/dense_odevit_dim768_h12_nfe12.yaml` | resumes `last.ckpt` of an interrupted run | set `setup.experiment.recover_from_checkpoint: false` |
| `ssog/baselines/imagenet/vit_b16_resume_e31_lr2e4.yaml` | continuation stage of an earlier run | run after the previous stage; keep as is |
| `ssog/baselines/imagenet/vit_b16_resume_raw_lr2e4.yaml` | continuation stage of an earlier run | run after the previous stage; keep as is |
| `ssog/cifar100/fixed_dim768_h6_r8_nfe12.yaml` | reattaches to W&B run `sbrxo4o8` (`resume: must`) | delete `setup.wandb.id` and `setup.wandb.resume`; set `recover_from_checkpoint: false` |
| `ssog/cifar100/fixed_r2_nfe12.yaml` | reattaches to W&B run `0vuhh1a3` (`resume: must`) | delete `setup.wandb.id` and `setup.wandb.resume`; set `recover_from_checkpoint: false` |
| `ssog/cifar100/fixed_r4_nfe16.yaml` | reattaches to W&B run `zodmg0b9` (`resume: must`) | delete `setup.wandb.id` and `setup.wandb.resume`; set `recover_from_checkpoint: false` |
| `ssog/cifar100/fixed_r4_nfe6.yaml` | reattaches to W&B run `l1621fgl` (`resume: must`) | delete `setup.wandb.id` and `setup.wandb.resume`; set `recover_from_checkpoint: false` |
| `ssog/cifar100/steered_r4_nfe12.yaml` | reattaches to W&B run `0qw2xu39` (`resume: must`) | delete `setup.wandb.id` and `setup.wandb.resume`; set `recover_from_checkpoint: false` |
| `ssog/imagenet/fixed_r4_nfe12_resume.yaml` | reattaches to W&B run `ofnpnpbt` (`resume: allow`) | delete `setup.wandb.id` and `setup.wandb.resume`; set `recover_from_checkpoint: false` |
| `ssog/looped_ssog/imagenet/fixed_dim768_h12_r4_depth12.yaml` | resumes `last-v2.ckpt` of an interrupted run | set `setup.experiment.recover_from_checkpoint: false` |
