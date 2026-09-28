"""GPU inference benchmark for SSOG-ODEViT and transformer baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf
from torchvision.models.vision_transformer import VisionTransformer

from models.ODEViT import ODE_ViT
from models.SSOGViT import SSOG_ViT

from models.MatchedViT import MatchedBViT
from models.DiscreteSSOGViT import DiscreteSSOG_ViT

TRANSFORMER_SPECS = {
    "vit_tiny": {"dim": 192, "depth": 12, "heads": 3},
    "vit_small": {"dim": 384, "depth": 12, "heads": 6},
    "vit_base": {"dim": 768, "depth": 12, "heads": 12},
    "vit_large": {"dim": 1024, "depth": 24, "heads": 16},
    "deit_tiny": {"dim": 192, "depth": 12, "heads": 3},
    "deit_small": {"dim": 384, "depth": 12, "heads": 6},
    "deit_base": {"dim": 768, "depth": 12, "heads": 12},
    # The original DeiT paper reports Tiny, Small, and Base. This Large entry
    # extrapolates the distilled DeiT design with conventional ViT-L geometry.
    "deit_large": {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "paper_variant": False,
    },
}
SSOG_SPECS = {
    "ssog_tiny": {"dim": 192, "heads": 3, "num_eval_steps": 13},
    "ssog_small": {"dim": 384, "heads": 6, "num_eval_steps": 13},
    "ssog_medium": {"dim": 576, "heads": 9, "num_eval_steps": 13},
    "ssog_base": {"dim": 768, "heads": 12, "num_eval_steps": 13},
    "ssog_large": {"dim": 1024, "heads": 16, "num_eval_steps": 13},
    # Depth-matched control for the 24-layer ViT/DeiT-Large models.
    "ssog_large_nfe24": {
        "dim": 1024, "heads": 16, "num_eval_steps": 25,
        "paper_variant": False,
    },
}
PAIRED_SIZE_SPECS = {
    "tiny": {"dim": 192, "heads": 3, "depth": 12},
    "small": {"dim": 384, "heads": 6, "depth": 12},
    "medium": {"dim": 576, "heads": 9, "depth": 12},
    "base": {"dim": 768, "heads": 12, "depth": 12},
    "large": {"dim": 1024, "heads": 16, "depth": 24},
}
PAIRED_FAMILIES = ("odevit", "vit", "bvit", "deit")
PAIRED_MIXERS = ("dense", "ssog")
PAIRED_MODEL_NAMES = tuple(
    f"{family}_{size}_{mixer}"
    for family in PAIRED_FAMILIES
    for size in PAIRED_SIZE_SPECS
    for mixer in PAIRED_MIXERS
)
MODEL_NAMES = (
    "ssog", "odevit", "vit", *SSOG_SPECS, *TRANSFORMER_SPECS,
    *PAIRED_MODEL_NAMES,
)


class DistilledVisionTransformer(VisionTransformer):
    """DeiT-style ViT with a distillation token and a second classifier head."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distillation_token = nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.distillation_token, std=0.02)

        # torchvision's encoder owns the positional embedding. DeiT requires
        # one additional position for its distillation token.
        self.seq_length += 1
        self.encoder.pos_embedding = nn.Parameter(
            torch.empty(1, self.seq_length, self.hidden_dim).normal_(std=0.02)
        )
        self.distillation_head = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.zeros_(self.distillation_head.weight)
        nn.init.zeros_(self.distillation_head.bias)

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        distillation_token = self.distillation_token.expand(
            batch_size, -1, -1
        )
        x = torch.cat([class_token, distillation_token, x], dim=1)
        x = self.encoder(x)
        class_logits = self.heads(x[:, 0])
        distillation_logits = self.distillation_head(x[:, 1])
        return (class_logits + distillation_logits) / 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/benchmarks/efficiency_resolution.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--resolutions", nargs="+", type=int, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    return parser.parse_args()


def paired_name_parts(name: str):
    parts = name.split("_")
    if (len(parts) == 3 and parts[0] in PAIRED_FAMILIES
            and parts[1] in PAIRED_SIZE_SPECS and parts[2] in PAIRED_MIXERS):
        return tuple(parts)
    return None


def build_paired_model(name: str, resolution: int, cfg):
    family, size, mixer = paired_name_parts(name)
    spec = PAIRED_SIZE_SPECS[size]
    patch_size = int(cfg.model.patch_size)
    mlp_ratio = float(cfg.model.mlp_ratio)
    num_classes = int(cfg.model.num_classes)
    common = dict(img_size=resolution, patch_size=patch_size, dim=spec["dim"],
                  num_heads=spec["heads"], mlp_ratio=mlp_ratio,
                  num_classes=num_classes)
    if family == "odevit":
        ode_common = dict(**common, num_eval_steps=spec["depth"] + 1,
                          solver=str(cfg.model.solver), regularize=False,
                          return_error_upper_bound=False, capture_attention=False)
        if mixer == "ssog":
            return SSOG_ViT(**ode_common, num_atoms=int(cfg.model.num_atoms),
                            lookat=bool(cfg.model.ssog_lookat),
                            sigma_floor=float(cfg.model.sigma_floor))
        return ODE_ViT(**ode_common,
                       gated_attention=bool(cfg.model.odevit_gated_attention))
    if family == "bvit" and mixer == "dense":
        return MatchedBViT(img_size=resolution, patch_size=patch_size,
                           dim=spec["dim"], num_heads=spec["heads"],
                           num_classes=num_classes, num_steps=spec["depth"],
                           mlp_ratio=mlp_ratio, dropout=0.0,
                           attention_dropout=0.0)
    if mixer == "ssog":
        return DiscreteSSOG_ViT(
            img_size=resolution, patch_size=patch_size, dim=spec["dim"],
            num_heads=spec["heads"], mlp_ratio=mlp_ratio,
            num_classes=num_classes, depth=spec["depth"],
            num_atoms=int(cfg.model.num_atoms),
            lookat=bool(cfg.model.ssog_lookat),
            sigma_floor=float(cfg.model.sigma_floor),
            recurrent=family == "bvit", distilled=family == "deit",
            capture_attention=False)
    model_class = DistilledVisionTransformer if family == "deit" else VisionTransformer
    return model_class(image_size=resolution, patch_size=patch_size,
                       num_layers=spec["depth"], num_heads=spec["heads"],
                       hidden_dim=spec["dim"], mlp_dim=int(spec["dim"] * mlp_ratio),
                       dropout=0.0, attention_dropout=0.0, num_classes=num_classes)


def build_model(name: str, resolution: int, cfg):
    if paired_name_parts(name):
        return build_paired_model(name, resolution, cfg)
    ssog_spec = SSOG_SPECS.get(name)
    common = dict(
        img_size=resolution,
        patch_size=int(cfg.model.patch_size),
        dim=int(ssog_spec["dim"] if ssog_spec else cfg.model.dim),
        num_heads=int(ssog_spec["heads"] if ssog_spec else cfg.model.num_heads),
        mlp_ratio=float(cfg.model.mlp_ratio),
        num_classes=int(cfg.model.num_classes),
    )
    if name == "ssog" or name in SSOG_SPECS:
        return SSOG_ViT(
            **common,
            num_eval_steps=int(
                ssog_spec.get("num_eval_steps", cfg.model.num_eval_steps)
                if ssog_spec else cfg.model.num_eval_steps
            ),
            solver=str(cfg.model.solver),
            regularize=False,
            return_error_upper_bound=False,
            num_atoms=int(cfg.model.num_atoms),
            lookat=bool(cfg.model.ssog_lookat),
            sigma_floor=float(cfg.model.sigma_floor),
            capture_attention=False,
        )
    if name == "odevit":
        return ODE_ViT(
            **common,
            num_eval_steps=int(cfg.model.num_eval_steps),
            solver=str(cfg.model.solver),
            regularize=False,
            return_error_upper_bound=False,
            gated_attention=bool(cfg.model.odevit_gated_attention),
            capture_attention=False,
        )
    if name == "vit":
        return VisionTransformer(
            image_size=resolution,
            patch_size=int(cfg.model.patch_size),
            num_layers=int(cfg.model.vit_depth),
            num_heads=int(cfg.model.num_heads),
            hidden_dim=int(cfg.model.dim),
            mlp_dim=int(cfg.model.dim * cfg.model.mlp_ratio),
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=int(cfg.model.num_classes),
        )
    if name in TRANSFORMER_SPECS:
        spec = TRANSFORMER_SPECS[name]
        model_class = (
            DistilledVisionTransformer
            if name.startswith("deit_")
            else VisionTransformer
        )
        return model_class(
            image_size=resolution,
            patch_size=int(cfg.model.patch_size),
            num_layers=spec["depth"],
            num_heads=spec["heads"],
            hidden_dim=spec["dim"],
            mlp_dim=int(spec["dim"] * cfg.model.mlp_ratio),
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=int(cfg.model.num_classes),
        )
    raise ValueError(f"Unknown model: {name}")


def autocast_context(dtype: str):
    if dtype == "fp32":
        return nullcontext()
    amp_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def forward(model, name: str, images: torch.Tensor):
    paired = paired_name_parts(name)
    is_ode = name in ("ssog", "odevit") or name in SSOG_SPECS
    is_ode = is_ode or (paired is not None and paired[0] == "odevit")
    if is_ode:
        result = model(images, output_attentions=False,
                       output_attentions_trajectory=False,
                       return_error_upper_bound=False, regularize=False)
    else:
        result = model(images)
    return result.logits if hasattr(result, "logits") else result


def profile_flops(model, name: str, images: torch.Tensor, dtype: str):
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        with torch.inference_mode(), autocast_context(dtype):
            forward(model, name, images)
        torch.cuda.synchronize()
    flops = sum(event.flops or 0 for event in prof.key_averages())
    return int(flops) if flops else None


def token_count(name: str, resolution: int, patch_size: int) -> int:
    paired = paired_name_parts(name)
    special_tokens = 2 if name.startswith("deit_") else 1
    if paired is not None and paired[0] == "deit":
        special_tokens = 2
    return (resolution // patch_size) ** 2 + special_tokens


def model_metadata(name: str, cfg) -> dict:
    paired = paired_name_parts(name)
    if paired is not None:
        family, size, mixer = paired
        spec = PAIRED_SIZE_SPECS[size]
        return {"model_family": family, "model_size": size,
                "model_mixer": mixer, "comparison_key": f"{family}_{size}",
                "model_dim": spec["dim"], "model_depth": spec["depth"],
                "model_heads": spec["heads"],
                "paper_variant": not (family == "deit" and size == "large")}
    if name in SSOG_SPECS:
        spec = SSOG_SPECS[name]
        return {
            "model_family": "ssog",
            "model_dim": spec["dim"],
            "model_depth": spec["num_eval_steps"] - 1,
            "model_heads": spec["heads"],
            "paper_variant": spec.get("paper_variant", True),
        }
    if name in TRANSFORMER_SPECS:
        spec = TRANSFORMER_SPECS[name]
        return {
            "model_family": name.split("_", 1)[0],
            "model_dim": spec["dim"],
            "model_depth": spec["depth"],
            "model_heads": spec["heads"],
            "paper_variant": spec.get("paper_variant", True),
        }
    return {
        "model_family": name,
        "model_dim": int(cfg.model.dim),
        "model_depth": (
            int(cfg.model.vit_depth) if name == "vit" else None
        ),
        "model_heads": int(cfg.model.num_heads),
        "paper_variant": True,
    }


def benchmark_case(model, name: str, resolution: int, batch_size: int, cfg,
                   profile_flops_enabled: bool = True):
    device = torch.device("cuda")
    dtype = str(cfg.benchmark.dtype)
    images = torch.randn(batch_size, 3, resolution, resolution, device=device)

    with torch.inference_mode():
        for _ in range(int(cfg.benchmark.warmup_iterations)):
            with autocast_context(dtype):
                forward(model, name, images)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms = []
    with torch.inference_mode():
        for _ in range(int(cfg.benchmark.measure_iterations)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_context(dtype):
                forward(model, name, images)
            end.record()
            end.synchronize()
            latencies_ms.append(start.elapsed_time(end))

    mean_ms = statistics.fmean(latencies_ms)
    sorted_ms = sorted(latencies_ms)
    p50_ms = statistics.median(sorted_ms)
    p95_index = min(len(sorted_ms) - 1, math.ceil(0.95 * len(sorted_ms)) - 1)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

    flops = None
    if profile_flops_enabled:
        flops = profile_flops(model, name, images, dtype)

    return {
        **model_metadata(name, cfg),
        "model": name,
        "resolution": resolution,
        "tokens": token_count(name, resolution, int(cfg.model.patch_size)),
        "batch_size": batch_size,
        "dtype": dtype,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": sorted_ms[p95_index],
        "images_per_second": batch_size * 1000.0 / mean_ms,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "profiler_flops_per_batch": flops,
        "profiler_gflops_per_image": (
            flops / batch_size / 1.0e9 if flops is not None else None
        ),
        "status": "ok",
    }


def failure_row(name, resolution, batch_size, cfg, error):
    return {
        **model_metadata(name, cfg),
        "model": name,
        "resolution": resolution,
        "tokens": token_count(name, resolution, int(cfg.model.patch_size)),
        "batch_size": batch_size,
        "dtype": str(cfg.benchmark.dtype),
        "status": "oom" if "out of memory" in str(error).lower() else "error",
        "error": str(error).replace("\n", " ")[:500],
    }


def write_results(rows, output_dir: Path, cfg):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "resolution_scaling.csv"
    json_path = output_dir / "resolution_scaling.json"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "flops_note": "torch.profiler operator FLOPs; unsupported operators may be omitted",
            "model_notes": {
                "deit_tiny": "Original DeiT distilled Tiny geometry.",
                "deit_small": "Original DeiT distilled Small geometry.",
                "deit_base": "Original DeiT distilled Base geometry.",
                "deit_large": (
                    "Extrapolated distilled DeiT with ViT-L/16 geometry; "
                    "not an original DeiT paper configuration."
                ),
            },
        },
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")

    cfg = OmegaConf.load(args.config)
    models = args.models or list(cfg.benchmark.models)
    resolutions = args.resolutions or list(cfg.benchmark.resolutions)
    batch_sizes = args.batch_sizes or list(cfg.benchmark.batch_sizes)
    output_dir = Path(args.output_dir or cfg.output.directory)
    repetitions = int(cfg.benchmark.get("repetitions", 1))
    profile_batches = set(int(value) for value in cfg.benchmark.get(
        "profile_flops_batch_sizes", [batch_sizes[0]]))
    rows = []

    for name in models:
        for resolution in resolutions:
            model = None
            try:
                model = build_model(name, int(resolution), cfg).eval().cuda()
                for batch_size in batch_sizes:
                    for repeat in range(repetitions):
                        try:
                            row = benchmark_case(
                                model, name, int(resolution), int(batch_size), cfg,
                                profile_flops_enabled=(bool(cfg.benchmark.profile_flops)
                                    and repeat == 0 and int(batch_size) in profile_batches))
                        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                            row = failure_row(name, int(resolution), int(batch_size), cfg, error)
                            torch.cuda.empty_cache()
                        row["repeat"] = repeat
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        if row["status"] == "oom":
                            break
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                for batch_size in batch_sizes:
                    rows.append(
                        failure_row(name, int(resolution), int(batch_size), cfg, error)
                    )
            finally:
                del model
                torch.cuda.empty_cache()

    csv_path, json_path = write_results(rows, output_dir, cfg)
    print(f"CSV results: {csv_path}")
    print(f"JSON results: {json_path}")


if __name__ == "__main__":
    main()
