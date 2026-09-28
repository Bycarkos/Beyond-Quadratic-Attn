# SSOG publication experiment suite

These configs form a controlled experiment matrix for evaluating separable
sums of Gaussians in ODE-ViT. They do not by themselves prove publishability;
the paper also needs competitive baselines, repeated seeds, uncertainty, and
compute/parameter reporting.

## Run order

1. Main CIFAR-100 result: `cifar100/fixed_r4_nfe12.yaml`.
2. Atom ablation: fixed R1, R2, R4, and R8 at NFE 12.
3. ODE-depth ablation: fixed R4 at NFE 6, 8, 12, and 16.
4. Steering ablation: steered R4 at NFE 6, 8, and 12. Fixed and steered
   comparisons must use equal NFE.
5. Mechanism ablations: steered cold versus warm initialization, fixed sigma
   floors 0.10/0.25/0.50, and steered maximum offsets 2/4/7. The steered
   R4/NFE12 config is the cold control; `steered_r4_warm_init.yaml` changes
   only `cold_init`.
6. ImageNet confirmation: fixed R2/R4/R8 and steered R4 at NFE 12. Run this
   stage only after choosing the CIFAR-100 candidate.

The unmodified base value is sigma floor 0.25, cold initialization, and
maximum offset 7. The baseline fixed R4/NFE12 config therefore supplies the
middle point for each corresponding ablation and should not be rerun merely
under another filename.

## Launching

From the repository root, pass the path relative to `configs/classification`:

```bash
python main_classification_ssog.py -c ssog/cifar100/fixed_r4_nfe12.yaml
python main_classification_ssog.py -c ssog/imagenet/fixed_r4_nfe12.yaml
```

`num_eval_steps` counts evaluation points. Euler uses one fewer transitions,
so values 7, 9, 13, and 17 correspond to logged NFE 6, 8, 12, and 16.

Each run has an isolated checkpoint directory based on its W&B name. This is
required when several jobs execute concurrently and prevents the checkpoint
collisions seen in the exploratory runs.

## Augmentation ablation

The controlled scratch-training matrix is under `augmentation/{cifar100,imagenet}`.
For each dataset it crosses fixed and fully steered SSOG with three levels:

| Level | Crop scale | RandAugment | Random erasing | Mixup | CutMix | Label smoothing |
|---|---:|---:|---:|---:|---:|---:|
| Weak | 0.8--1.0 | off | 0 | 0 | 0 | 0 |
| Moderate | 0.7--1.0 | 2 ops, magnitude 5 | 0.10 | 0.10 | 0.50 | 0.05 |
| Strong | 0.4--1.0 | 2 ops, magnitude 9 | 0.25 | 0.20 | 1.00 | 0.10 |

The strong setting reproduces the original scratch-training augmentation. All
other model and optimization settings come from the dataset base config. Fixed
and steered pairs differ only in `modeling.inputs.lookat`. Every run has
`recover_from_checkpoint: false`, seed 42, and a unique W&B/checkpoint name.

Launch a CIFAR-100 run with:

```bash
sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh \
  ssog/augmentation/cifar100/fixed_weak.yaml
```

Launch an ImageNet run with:

```bash
sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh \
  ssog/augmentation/imagenet/fixed_weak.yaml
```

Replace `fixed_weak` with `fixed_moderate`, `fixed_strong`,
`steered_weak`, `steered_moderate`, or `steered_strong` for the other cells.

## Required paper protocol

- Report validation top-1/top-3/top-5, loss, NFE, parameters, wall-clock
  training time, inference throughput, peak GPU memory, and mean +/- standard
  deviation over at least three seeds.
- Compare against the repository's standard ODE-ViT using the same data,
  augmentation, optimizer, schedule, dimension, effective batch, and NFE.
- Include a standard ViT/attention baseline matched as closely as possible in
  parameters and training compute.
- Keep the four qualitative validation images and queries fixed across all
  variants. Report sigma separately for x and y before making claims about
  anisotropy.
- Use a held-out test set or a predeclared final validation protocol; do not
  choose hyperparameters from the test set.

## Reproducibility

`main_classification_scratch.py` seeds Python, NumPy, PyTorch, workers, model
initialization, and dataloader construction before creating those objects. The
augmentation configs declare seed 42 explicitly. For paper statistics, repeat
the selected weak/moderate/strong comparison with at least three seeds and
report mean and standard deviation.

## Discrete SSOG baseline

The independent 12-block CIFAR-100 baseline is documented in
[discrete/README.md](discrete/README.md). Launch it with
`sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh <config>`. It reuses the SSOG spatial
operator and training recipe, with patch-mean readout and no ODE solver.
