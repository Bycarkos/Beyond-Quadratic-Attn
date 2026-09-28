"""Isolate the attention operator's own memory footprint, stripped of every
confound (parameter count, patch embedding, MLP, classifier head, model
class). Compares SSOGAttention (fixed/non-lookat) against exactly the dense
attention path used by dense baselines in this codebase
(torch.nn.MultiheadAttention with need_weights=False, i.e. PyTorch's fused
SDPA fast path -- see torchvision.models.vision_transformer.EncoderBlock).

Both modules receive identical [B, N, D] input at matching token counts N,
with everything else (dim, heads, batch) held fixed. Peak memory is reset
and captured around ONLY the attention module's own forward call.
"""

import torch
import torch.nn as nn

from models.SSOGViT import SSOGAttention

DIM = 768
HEADS = 12
BATCH = 8
NUM_ATOMS = 4
SIGMA_FLOOR = 0.25
PATCH_SIZE = 16

RESOLUTIONS = [224, 384, 512, 640, 768, 1024]


def grid_size(resolution: int) -> int:
    return resolution // PATCH_SIZE


def measure(module: nn.Module, x: torch.Tensor, is_ssog: bool) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(20):
            if is_ssog:
                module(x)
            else:
                module(x, x, x, need_weights=False)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main():
    device = torch.device("cuda")
    print(f"{'N':>6} {'tokens':>7} {'ssog_mem_mib':>13} {'dense_mem_mib':>14} {'ssog/dense':>11}")

    for resolution in RESOLUTIONS:
        grid_h = grid_w = grid_size(resolution)
        num_patches = grid_h * grid_w
        num_tokens = num_patches + 1  # + CLS

        x = torch.randn(BATCH, num_tokens, DIM, device=device, dtype=torch.float16)

        ssog = SSOGAttention(
            dim=DIM, num_heads=HEADS, grid_h=grid_h, grid_w=grid_w,
            num_atoms=NUM_ATOMS, lookat=False, sigma_floor=SIGMA_FLOOR,
            has_cls_token=True, capture_attention=False,
        ).to(device=device, dtype=torch.float16).eval()

        dense = nn.MultiheadAttention(
            DIM, HEADS, batch_first=True,
        ).to(device=device, dtype=torch.float16).eval()

        ssog_mem = measure(ssog, x, is_ssog=True)
        dense_mem = measure(dense, x, is_ssog=False)

        print(f"{resolution:>6} {num_tokens:>7} {ssog_mem:>13.2f} {dense_mem:>14.2f} {ssog_mem / dense_mem:>11.3f}")

        del ssog, dense, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
