#!/usr/bin/env python3
"""FLOPs of SSoG vs dense (SDPA) attention for the three architecture settings.

Uses the same six dim768 models and resolutions as the isolated efficiency
benchmarks (benchmark_efficiency_resolution.build_model). Counts are
per-image forward FLOPs (2 per multiply-add) from torch's FlopCounterMode, for
the whole model and for the attention modules alone (including their
projections). Element-wise ops (softmax, norms, activations) are not counted,
for either operator.

Dense attention is counted through its reference path: the MHA fast path is
disabled and SDPA is forced to the MATH backend, because the counter has no
formula for the fused kernels and would silently drop the attention FLOPs.
The arithmetic is identical; only the kernel differs. Verified against the
analytic ViT-B/16 count (35.13 GFLOPs at 224px, 100 classes).
"""

import csv
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from assets.benchmarking import benchmark_efficiency_resolution as B

CONFIG = "configs/benchmarks/efficiency_vit_ssog_isolated.yaml"
OUTPUT = Path("experiments/SSOG/efficiency/flops/flops.csv")
FAMILIES = ("odevit", "vit", "bvit")
MIXERS = ("dense", "ssog")
RESOLUTIONS = (224, 384, 512, 640, 768, 1024)
# Top-level attention modules (children such as .v_proj are included in them).
ATTENTION_SUFFIXES = (".self_attention", ".attn")


def count(name, resolution, cfg):
    model = B.build_model(name, resolution, cfg).eval()
    images = torch.randn(1, 3, resolution, resolution)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), \
            FlopCounterMode(display=False, depth=None) as counter:
        B.forward(model, name, images)
    per_module = counter.get_flop_counts()
    # Counter keys are "<ModelClass>.<module path>". The continuous models'
    # shared ODE block is reachable under two paths, so deduplicate by module.
    modules = dict(model.named_modules(remove_duplicate=False))
    attention = {}
    for key, ops in per_module.items():
        path = key.split(".", 1)[1] if "." in key else ""
        if path.endswith(ATTENTION_SUFFIXES) and path in modules:
            attention[id(modules[path])] = sum(ops.values())
    return counter.get_total_flops(), sum(attention.values()), list(attention)


def main():
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = OmegaConf.load(CONFIG)
    rows = []
    for family in FAMILIES:
        for mixer in MIXERS:
            name = f"{family}_base_{mixer}"
            for resolution in RESOLUTIONS:
                total, attention, modules = count(name, resolution, cfg)
                rows.append(dict(family=family, mixer=mixer, resolution=resolution,
                                 tokens=B.token_count(name, resolution, int(cfg.model.patch_size)),
                                 total_gflops=round(total / 1e9, 3),
                                 attention_gflops=round(attention / 1e9, 3)))
                print(f"  {name:18s} {resolution:5d}px total {total/1e9:9.2f} GFLOPs  "
                      f"attention {attention/1e9:9.2f} ({len(modules)} attention modules)", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
