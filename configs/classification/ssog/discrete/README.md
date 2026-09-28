# Discrete SSOG CIFAR-100 baseline

Launch from the repository root:

```bash
sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh \
  ssog/discrete/cifar100/fixed_dim768_h12_r4_depth12.yaml
```

The same launcher runs the looped (recurrent) variant with a
`ssog/looped_ssog/cifar100/*.yaml` config.

On an allocated GPU, the Python entry point is:

```bash
python main_classification_discrete_ssog.py \
  -c ssog/discrete/cifar100/fixed_dim768_h12_r4_depth12.yaml
```

The launcher uses one L40S GPU and one task. The default experiment has 12
independently parameterized blocks, dimension 768, 12 heads, four fixed Gaussian
atoms per head, 224-pixel inputs, and 16-pixel patches. Each block applies
`x = x + SSOG(CenterNorm(x))`, then `x = x + MLP(CenterNorm(x))`. It reuses the
existing SSOGAttention and ELU MLP, with the ODE SSOG spectral initialization
and 0.1 output-projection scaling. It has no TimeFiLM, solver, time gates, or
integration-step scaling. Depth is logged as `train/depth` and `val/depth`,
not NFE.

The model has patch tokens only and classifies their final global average.
Reusing ODE SSOG's CLS-only classifier would leave the final block's Gaussian
geometry without a classification-loss gradient, because its CLS path averages
values before spatial filtering. The tests verify nonzero gradients through
every block's centers, widths, mixtures, and MLP. Accordingly, this is a
conventional independent-block baseline, not an isolation of weight sharing
alone: block composition, time conditioning, and readout also differ from ODE
SSOG. A later tied/untied Euler control would isolate sharing more closely.

The recipe matches the existing fixed-R4 CIFAR ODE control: 500 epochs, seed 42,
Muon/auxiliary optimizer, cosine schedule, augmentation, batch 64 with
accumulation 2 (effective batch 128), and validation batch 64. The discrete
base has an explicit model schema without ODE-only kwargs; a configuration
parity test checks that the shared data/optimization settings do not drift.
Training uses cross-entropy with the inherited mixed/smoothed targets. ODE
regularizers are absent, matching the effective loss of the fixed SSOG control
with attention capture disabled.

The shared trainer retains classification metrics and fixed validation-image
visualizations. All `ssog_*` statistics and atom figures refer to the **last
block only**, not the whole network. Attention capture is disabled in training;
visualizations use the existing normalized-input hook and per-query kernels.
Full hidden-state storage is opt-in in the discrete forward pass.

W&B name: `cifar100-discrete-dim768-h12-r4-depth12-seed42` in `SSOG-Paper/cifar100`.
Checkpoints are isolated under `~/checkpoints/SSOG-Paper/cifar100/` using that
name. For later seeds, create a config with a different seed and W&B name; do
not run simultaneous jobs with the same checkpoint directory. The shared
`recover_from_checkpoint` and `resume_checkpoint_path` settings work normally.

This inherits the historical pipeline's use of the official CIFAR test split
as `val`. It does not introduce the prospective train/validation split discussed
in the paper review. Report this protocol honestly; use a separate split for
new hyperparameter selection. Independent blocks also increase parameters:
report measured counts and costs rather than treating equal width/depth as
parameter matching.

## Checks

Run on a compute node:

```bash
python -m unittest tests.ssog_experiments.test_discrete_ssog \
  tests.ssog_experiments.test_ssog_attention -v
```

The opt-in full-size smoke test executes the actual shared training pipeline
for one training and validation batch, with W&B/checkpoint writes disabled:

```bash
python -m tests.ssog_experiments.smoke_discrete_ssog
```

Validated 2026-09-08: all eight discrete/operator unit tests passed on a Slurm
compute node. The full publication configuration completed one real CIFAR-100
training/optimizer step and one validation batch on an L40S, using 10.09 GiB
peak CUDA allocated memory. Lightning reported 71.6M trainable parameters.
These are smoke-test measurements, not a publication efficiency benchmark.
The 500-epoch training job was not submitted.
