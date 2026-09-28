# assets/benchmarking

Speed, memory and FLOP measurements. Everything except `count_flops.py` needs a
CUDA GPU. All measurements are **inference** (forward pass under
`torch.inference_mode()`, models in `eval()`), not training cost. Run from the
repository root.

## benchmark_efficiency_resolution.py

The main efficiency benchmark behind the paper's throughput and peak-memory
numbers. For each model, resolution and batch size in a config it measures
images per second and peak allocated memory.

**Run**

```bash
sbatch scripts/supervised/benchmarking/benchmark_efficiency_resolution.sh configs/benchmarks/efficiency_vit_ssog_isolated.yaml
python -m assets.benchmarking.benchmark_efficiency_resolution --config configs/benchmarks/efficiency_vit_ssog_isolated.yaml
```

| Option | Default | |
|---|---|---|
| `--config` | `configs/benchmarks/efficiency_resolution.yaml` | models, resolutions, batches, dtype, iterations, output directory |
| `--output-dir` | the config's `output.directory` | |
| `--models`, `--resolutions`, `--batch-sizes` | from the config | override a subset |

**Important:** measure one model per process. The paper uses the six
`configs/benchmarks/efficiency_{odevit,vit,bvit}_{dense,ssog}_isolated.yaml`
configs, one job each. Benchmarking several models in one process
contaminated memory and throughput readings in an earlier sweep.

**Writes** `resolution_scaling.csv` and `.json` in the output directory
(one row per model × resolution × batch, with `images_per_second`,
`peak_allocated_mib` and a `status` column).

**Expected results:** about 15–30 minutes per isolated config on one A40
(six resolutions × seven batch sizes). The resulting CSVs feed
`plotting/plot_efficiency_delta_by_resolution.py`,
`plotting/plot_efficiency_dual_axis*.py` and
`tables/generate_delta_by_resolution_table.py`.

## count_flops.py

Forward FLOPs per image (2 per multiply-add) of the six dim768 benchmark models,
for the whole model and for the attention modules alone, at 224–1024 px.
Counted with `torch.utils.flop_counter.FlopCounterMode`. Dense attention is
counted through its reference path (MHA fast path off, SDPA forced to the MATH
backend), because the counter has no formula for the fused kernels and would
silently drop the attention FLOPs.

**Run** (CPU, no GPU needed; several minutes)

```bash
python -m assets.benchmarking.count_flops
```

**Writes** `experiments/SSOG/efficiency/flops/flops.csv`.

**Expected results:** the dense discrete model at 224 px is 35.13 GFLOPs,
matching the analytic ViT-B/16 count. SSoG saves 19% of whole-model FLOPs at
224 px and 53% at 1024 px; 53–82% for the attention modules. The three
settings agree to within 0.3 GFLOPs.

## benchmark_attention_isolated.py

Peak memory of the attention operator alone (SSoG vs `nn.MultiheadAttention`
on its fused SDPA path), with identical inputs at matching token counts, to
separate the operator from the rest of the network.

**Run**

```bash
python -m assets.benchmarking.benchmark_attention_isolated
```

**Writes** nothing; prints one line per resolution
(tokens, SSoG MiB, dense MiB, ratio).

**Expected results:** SSoG uses about 1.6–2.3× the memory of the fused dense
kernel at batch 8 (the finding behind the paper's operator-level memory
appendix).

## benchmark_euler_rollout.py

Compares the hand-written fixed-step Euler loop of `SSOG_ViT` with
`torchdiffeq.odeint(..., method="euler")`: first a correctness check (same
logits, hidden states and gradients), then forward/backward timing.

```bash
sbatch scripts/supervised/benchmarking/benchmark_euler_rollout.sh
python -m assets.benchmarking.benchmark_euler_rollout --batch-size 64 --steps 20
```

Options: `--dim --heads --atoms --num-eval-steps --img-size --patch-size
--batch-size --num-classes --steps --warmup` (defaults: the dim768 CIFAR-100
model). Prints the check result and timings; writes nothing.

## benchmark_fixed_kernel_cache.py

Checks and times two `SSOG_ViT` optimizations: computing the fixed kernel once
per forward pass instead of once per ODE step, and batching the time
conditioning (TimeFiLM) over the whole time grid.

```bash
python -m assets.benchmarking.benchmark_fixed_kernel_cache --batch-size 64
```

Same options as `benchmark_euler_rollout.py`. Prints results; writes nothing.
This script has no Slurm launcher.

## audit_ssog_apply_bottleneck.py

Profiles where forward and backward time goes inside fixed `SSOG_ViT`, to decide
whether a fused kernel for the Gaussian "apply" step is worth writing. It only
wraps methods of one throwaway model instance with timers; no code is modified.

```bash
sbatch scripts/supervised/benchmarking/audit_ssog_apply_bottleneck.sh --profile
python -m assets.benchmarking.audit_ssog_apply_bottleneck --batch-size 64 --profile
```

Options as above, plus `--profile`, `--profile-regions`, `--trace-path`
(default `ssog_trace.json`) and `--regions-trace-path`. Prints a time
breakdown; with `--profile` it also writes Chrome traces (open them in
`chrome://tracing` or Perfetto).
