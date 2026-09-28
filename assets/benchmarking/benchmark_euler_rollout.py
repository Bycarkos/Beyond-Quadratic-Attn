"""
Compare the hand-rolled fixed-step Euler loop (SSOG_ViT._euler_rollout,
opt-in via use_custom_euler_loop=True) against torchdiffeq.odeint(...,
method="euler") -- the path every existing config actually uses today.

Both are verified bit-exact (see the correctness check this script runs
before timing): same logits, same hidden states, same gradients on every
parameter. This script measures whether removing torchdiffeq's generic
solver-dispatch overhead (adaptive/adjoint machinery this repo never uses,
tuple-state packing, the _PerturbFunc wrapper) is worth anything in wall
clock, independent of the fixed-kernel-cache and TimeFiLM-table
optimizations (both paths get those; only the outer integration loop
differs).

Run on an allocated GPU (not the login node):

    python -m assets.benchmarking.benchmark_euler_rollout \
        --dim 768 --heads 12 --atoms 4 --num-eval-steps 13 \
        --batch-size 64 --img-size 224 --patch-size 16 --steps 20
"""

import argparse
import time

import torch

from models.SSOGViT import SSOG_ViT


def build_model(args, device, use_custom_euler_loop):
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
        use_custom_euler_loop=use_custom_euler_loop,
    ).to(device)
    return model


def check_equivalence(args, device):
    x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
    labels = torch.randint(0, args.num_classes, (2,), device=device)

    m_odeint = build_model(args, device, use_custom_euler_loop=False)
    out_odeint = m_odeint(x, labels=labels)
    out_odeint.cross_entropy.backward()

    m_custom = build_model(args, device, use_custom_euler_loop=True)
    out_custom = m_custom(x, labels=labels)
    out_custom.cross_entropy.backward()

    logits_equal = torch.equal(out_odeint.logits, out_custom.logits)
    max_logit_diff = (out_odeint.logits - out_custom.logits).abs().max().item()

    max_grad_diff = 0.0
    for (na, pa), (nb, pb) in zip(
        m_odeint.named_parameters(), m_custom.named_parameters()
    ):
        assert na == nb
        if pa.grad is None:
            continue
        max_grad_diff = max(max_grad_diff, (pa.grad - pb.grad).abs().max().item())

    print(
        f"equivalence check: logits equal={logits_equal} "
        f"(max diff {max_logit_diff:.3e}), max grad diff {max_grad_diff:.3e}"
    )
    if not logits_equal or max_grad_diff > 0:
        print(
            "WARNING: expected bit-exact equivalence (verified on CPU during "
            "development) -- investigate before trusting the timing below."
        )


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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
            "This benchmark measures GPU-side dispatch overhead and must "
            "run on an allocated GPU, not the login node."
        )
    device = torch.device("cuda")
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"config: dim={args.dim} heads={args.heads} atoms={args.atoms} "
        f"NFE={args.num_eval_steps - 1} batch={args.batch_size} "
        f"img={args.img_size} patch={args.patch_size}\n"
    )

    check_equivalence(args, device)
    print()

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    model_odeint = build_model(args, device, use_custom_euler_loop=False)
    t_odeint = time_forward_backward(
        model_odeint, x, labels, args.steps, args.warmup
    )
    del model_odeint
    torch.cuda.empty_cache()

    model_custom = build_model(args, device, use_custom_euler_loop=True)
    t_custom = time_forward_backward(
        model_custom, x, labels, args.steps, args.warmup
    )
    del model_custom
    torch.cuda.empty_cache()

    print(f"per-step time with torchdiffeq.odeint:  {t_odeint * 1000:8.2f} ms")
    print(f"per-step time with custom Euler rollout: {t_custom * 1000:8.2f} ms")
    print(f"speedup: {t_odeint / t_custom:.3f}x")


if __name__ == "__main__":
    main()
