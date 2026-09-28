"""
Audit two SSOG_ViT micro-optimizations (see models/SSOGViT.py):

1. Fixed-kernel caching: fixed SSOG's geometric kernel (ay, ax, lam) depends
   only on learned parameters, not on the input tokens, so it is identical at
   every one of the solver's NFE evaluations within one forward pass.
   SSOG_ViT computes it once per forward pass instead of once per NFE step
   (SSOGAttention.set_fixed_kernel_override).

2. TimeFiLM table: TimeFiLM(t) only depends on the scalar time, and explicit
   fixed-step Euler evaluates the vector field at exactly the points of
   t_grid, so the whole grid is batched through the tiny time-conditioning
   MLP once instead of once per NFE step (Lie_Trotter.set_time_film_table).
   Guarded to solver="euler" only.

This script times forward+backward across three configurations built from
the same seed (identical weights) and the same input batch, isolating each
optimization's own contribution:

  baseline       - both optimizations defeated (pre-optimization behavior)
  kernel-only    - fixed-kernel caching on, TimeFiLM table defeated
  both (current) - current code, unmodified (both optimizations on)

Run on an allocated GPU (not the login node):

    python -m assets.benchmarking.benchmark_fixed_kernel_cache \
        --dim 768 --heads 12 --atoms 4 --num-eval-steps 13 \
        --batch-size 64 --img-size 224 --patch-size 16 --steps 20

`--num-eval-steps 13` corresponds to NFE 12 (Euler performs one fewer
transition than the number of grid points), matching the main publication
configs under configs/classification/ssog/cifar100/.
"""

import argparse
import time

import torch

from models.SSOGViT import SSOG_ViT


def build_model(args, device):
    torch.manual_seed(0)
    model = SSOG_ViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        dim=args.dim,
        num_heads=args.heads,
        mlp_ratio=4.0,
        num_eval_steps=args.num_eval_steps,
        num_atoms=args.atoms,
        lookat=False,
        num_classes=args.num_classes,
    ).to(device)
    return model


def defeat_kernel_cache(model):
    model.equation.attn.set_fixed_kernel_override = lambda kernel: None


def defeat_time_film_table(model):
    model.equation.set_time_film_table = lambda table: None


def time_forward_backward(model, x, labels, steps, warmup):
    device = x.device
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        out = model(x, labels=labels)
        out.cross_entropy.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        out = model(x, labels=labels)
        out.cross_entropy.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--num-eval-steps", type=int, default=13)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument(
        "--steps", type=int, default=20, help="timed forward+backward iterations"
    )
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark measures a GPU kernel-launch-overhead effect and "
            "must run on an allocated GPU, not the login node."
        )
    device = torch.device("cuda")
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"config: dim={args.dim} heads={args.heads} atoms={args.atoms} "
        f"NFE={args.num_eval_steps - 1} batch={args.batch_size} "
        f"img={args.img_size} patch={args.patch_size}\n"
    )

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    results = {}

    model = build_model(args, device)
    defeat_kernel_cache(model)
    defeat_time_film_table(model)
    results["baseline (both off)"] = time_forward_backward(
        model, x, labels, args.steps, args.warmup
    )
    del model
    torch.cuda.empty_cache()

    model = build_model(args, device)
    defeat_time_film_table(model)
    results["kernel cache only"] = time_forward_backward(
        model, x, labels, args.steps, args.warmup
    )
    del model
    torch.cuda.empty_cache()

    model = build_model(args, device)
    results["both (current code)"] = time_forward_backward(
        model, x, labels, args.steps, args.warmup
    )
    del model
    torch.cuda.empty_cache()

    baseline = results["baseline (both off)"]
    print(f"{'configuration':<22} {'ms/step':>10} {'speedup vs baseline':>20}")
    for name, t in results.items():
        print(f"{name:<22} {t * 1000:10.2f} {baseline / t:19.3f}x")


if __name__ == "__main__":
    main()
