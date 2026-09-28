"""
Audit where forward+backward time actually goes inside fixed SSOG_ViT, to
decide whether a fused Triton kernel for the Gaussian-mixture "apply" step
(the three einsums in SSOGAttention._fixed_apply) is worth writing.

This is a read-only audit. It does NOT modify models/SSOGViT.py, models/
layers.py, or any other file in the repository. It builds one throwaway
SSOG_ViT instance inside this process and temporarily wraps a handful of
that single instance's bound methods with CUDA-event timers that call
straight through to the original, unmodified implementation -- nothing is
reimplemented, so the numbers reflect the real code path. All patches are
restored before the script exits, and since they are instance-level
attribute assignments on a model object this script alone constructed, they
cannot leak into or affect any other process, model, or file.

Two independent measurements are produced:
  1. A component-level breakdown (kernel construction vs. the apply einsums
     vs. MLP vs. norms vs. TimeFiLM), via the monkeypatched CUDA-event
     timers described above.
  2. An optional (--profile) fully standard torch.profiler trace of the same
     forward+backward, with zero patching at all, as an independent
     cross-check -- exported as a Chrome trace you can open at
     chrome://tracing or https://ui.perfetto.dev.

Run on an allocated GPU (not the login node):

    python -m assets.benchmarking.audit_ssog_apply_bottleneck \
        --dim 768 --heads 12 --atoms 4 --num-eval-steps 13 \
        --batch-size 64 --img-size 224 --patch-size 16 --steps 20

Add --profile to also dump a Chrome trace (ssog_trace.json by default).
"""

import argparse
import contextlib
import time
from collections import defaultdict

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


class ComponentTimer:
    """Accumulates per-region device time across many forward+backward
    calls. Uses torch.cuda.Event so it measures actual GPU execution time,
    not just Python-side dispatch latency."""

    def __init__(self, device):
        self.device = device
        self.totals_ms = defaultdict(float)
        self.counts = defaultdict(int)

    @contextlib.contextmanager
    def track(self, name):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            self.totals_ms[name] += start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            yield
            self.totals_ms[name] += (time.perf_counter() - t0) * 1000
        self.counts[name] += 1

    def report(self):
        print(f"\n{'component':<20} {'total ms':>10} {'calls':>7} {'ms/call':>10}")
        print("-" * 50)
        for name, total in sorted(self.totals_ms.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            print(f"{name:<20} {total:10.2f} {n:7d} {total / n:10.4f}")
        print(
            "\nNote: wrapping each region forces a GPU sync between them, so "
            "absolute per-call times are inflated relative to unbracketed "
            "execution (lost pipelining/overlap). The RELATIVE breakdown "
            "across components -- which one dominates -- is what this is for."
        )


def instrument(model, timer):
    """Temporarily wrap sub-steps of this one model instance with timing.
    Returns a callable that restores the original, unwrapped methods."""
    attn = model.equation.attn
    eq = model.equation

    originals = {
        "kernel_construct": attn._compute_fixed_kernel,
        "apply_einsums": attn._fixed_apply,
        "mlp": eq.mlp.forward,
        "time_film_percall": eq.time_film.forward,
        "time_film_grid": eq.time_film.forward_grid,
        "norm_attn": eq.norm_attn.forward,
        "norm_mlp": eq.norm_mlp.forward,
    }

    def make_wrapper(name, fn):
        def wrapped(*args, **kwargs):
            with timer.track(name):
                return fn(*args, **kwargs)

        return wrapped

    attn._compute_fixed_kernel = make_wrapper(
        "kernel_construct", originals["kernel_construct"]
    )
    attn._fixed_apply = make_wrapper("apply_einsums", originals["apply_einsums"])
    eq.mlp.forward = make_wrapper("mlp", originals["mlp"])
    eq.time_film.forward = make_wrapper(
        "time_film_percall", originals["time_film_percall"]
    )
    eq.time_film.forward_grid = make_wrapper(
        "time_film_grid", originals["time_film_grid"]
    )
    eq.norm_attn.forward = make_wrapper("norm_attn", originals["norm_attn"])
    eq.norm_mlp.forward = make_wrapper("norm_mlp", originals["norm_mlp"])

    def restore():
        attn._compute_fixed_kernel = originals["kernel_construct"]
        attn._fixed_apply = originals["apply_einsums"]
        eq.mlp.forward = originals["mlp"]
        eq.time_film.forward = originals["time_film_percall"]
        eq.time_film.forward_grid = originals["time_film_grid"]
        eq.norm_attn.forward = originals["norm_attn"]
        eq.norm_mlp.forward = originals["norm_mlp"]

    return restore


def run_component_breakdown(model, x, labels, steps, warmup, timer):
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        out = model(x, labels=labels)
        out.cross_entropy.backward()

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        with timer.track("__total_forward_backward__"):
            out = model(x, labels=labels)
            out.cross_entropy.backward()


def run_profiler_trace(model, x, labels, warmup, steps, trace_path):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        out = model(x, labels=labels)
        out.cross_entropy.backward()
    if x.device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            out = model(x, labels=labels)
            out.cross_entropy.backward()
        if x.device.type == "cuda":
            torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if x.device.type == "cuda" else "self_cpu_time_total"
    print("\n=== torch.profiler top operators (unmodified, unpatched code) ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace written to {trace_path} (open at chrome://tracing)")


def instrument_with_record_function(model):
    """Same idea as instrument() above -- temporarily wrap this one model
    instance's sub-steps, calling straight through to the original
    implementation -- but label each region with torch.profiler.
    record_function instead of a CUDA-event timer. record_function integrates
    with torch.profiler's own event stack, so its table can report a
    dedicated row per named region whose CPU/CUDA total already includes
    every op called inside it (e.g. the aten::bmm/aten::copy_ calls inside
    apply_einsums), rather than only a global, region-agnostic operator
    total. Forward-only, same caveat as ComponentTimer. Returns a restore
    callable."""
    attn = model.equation.attn
    eq = model.equation

    originals = {
        "kernel_construct": attn._compute_fixed_kernel,
        "apply_einsums": attn._fixed_apply,
        "mlp": eq.mlp.forward,
        "time_film_grid": eq.time_film.forward_grid,
        "norm_attn": eq.norm_attn.forward,
        "norm_mlp": eq.norm_mlp.forward,
    }

    def make_wrapper(name, fn):
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(f"ssog/{name}"):
                return fn(*args, **kwargs)

        return wrapped

    attn._compute_fixed_kernel = make_wrapper(
        "kernel_construct", originals["kernel_construct"]
    )
    attn._fixed_apply = make_wrapper("apply_einsums", originals["apply_einsums"])
    eq.mlp.forward = make_wrapper("mlp", originals["mlp"])
    eq.time_film.forward_grid = make_wrapper(
        "time_film_grid", originals["time_film_grid"]
    )
    eq.norm_attn.forward = make_wrapper("norm_attn", originals["norm_attn"])
    eq.norm_mlp.forward = make_wrapper("norm_mlp", originals["norm_mlp"])

    def restore():
        attn._compute_fixed_kernel = originals["kernel_construct"]
        attn._fixed_apply = originals["apply_einsums"]
        eq.mlp.forward = originals["mlp"]
        eq.time_film.forward_grid = originals["time_film_grid"]
        eq.norm_attn.forward = originals["norm_attn"]
        eq.norm_mlp.forward = originals["norm_mlp"]

    return restore


def run_profiler_trace_with_regions(model, x, labels, warmup, steps, trace_path):
    from torch.profiler import ProfilerActivity, profile

    restore = instrument_with_record_function(model)
    try:
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            out = model(x, labels=labels)
            out.cross_entropy.backward()
        if x.device.type == "cuda":
            torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU]
        if x.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True) as prof:
            for _ in range(steps):
                model.zero_grad(set_to_none=True)
                out = model(x, labels=labels)
                out.cross_entropy.backward()
            if x.device.type == "cuda":
                torch.cuda.synchronize()
    finally:
        restore()

    sort_key = "self_cuda_time_total" if x.device.type == "cuda" else "self_cpu_time_total"
    print("\n=== torch.profiler with named ssog/* regions (forward-only labels) ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))

    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace with named regions written to {trace_path}")


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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also run a standard, non-invasive torch.profiler pass and export a Chrome trace",
    )
    parser.add_argument(
        "--profile-regions",
        action="store_true",
        help=(
            "also run a torch.profiler pass with ssog/* region labels "
            "(record_function) around kernel_construct/apply_einsums/mlp/"
            "norms/time_film_grid, so aten::mm/aten::bmm/aten::copy_ time "
            "can be attributed to those regions instead of only a global total"
        ),
    )
    parser.add_argument("--trace-path", type=str, default="ssog_trace.json")
    parser.add_argument(
        "--regions-trace-path", type=str, default="ssog_trace_regions.json"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This audit measures GPU-side timing and must run on an "
            "allocated GPU, not the login node."
        )
    device = torch.device("cuda")
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"config: dim={args.dim} heads={args.heads} atoms={args.atoms} "
        f"NFE={args.num_eval_steps - 1} batch={args.batch_size} "
        f"img={args.img_size} patch={args.patch_size}"
    )

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    model = build_model(args, device)
    timer = ComponentTimer(device)
    restore = instrument(model, timer)
    try:
        run_component_breakdown(model, x, labels, args.steps, args.warmup, timer)
    finally:
        restore()
    timer.report()

    if args.profile:
        model2 = build_model(args, device)
        run_profiler_trace(model2, x, labels, args.warmup, args.steps, args.trace_path)

    if args.profile_regions:
        model3 = build_model(args, device)
        run_profiler_trace_with_regions(
            model3, x, labels, args.warmup, args.steps, args.regions_trace_path
        )


if __name__ == "__main__":
    main()
