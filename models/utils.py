import torch
from collections import defaultdict

from einops import rearrange

def pos_emb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature**omega)

    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def g_k(p, k=1):
    """f
    Compute g_k(p) = x_(k) * (1 - x_(k) + x_(k+1))
    where x_(k) and x_(k+1) are k-th and (k+1)-th largest elements per row.
    """
    sorted_p, _ = torch.sort(p, dim=-1, descending=True)
    x_k = sorted_p[..., k - 1]
    x_k1 = sorted_p[..., k] if k < p.size(-1) else torch.zeros_like(x_k)
    return x_k * (1 - x_k + x_k1)


def jasmin_loss(attn_maps, k=0, reduction="mean"):
    """
    Compute JaSMin regularization loss.

    attn_maps: [B, H, N, N] tensor or list of them.
    k=0 -> Eq.(8) from paper
    k>0 -> Eq.(9)
    """
    if isinstance(attn_maps, torch.Tensor):
        attn_maps = [attn_maps]

    losses = []
    for P in attn_maps:
        P = torch.clamp(P, min=1e-12, max=1.0)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
        g1 = g_k(P, k=1)
        if k == 0:
            loss = torch.log(g1 + 1e-12)
        else:
            gk = g_k(P, k=k)
            loss = torch.log((g1 / (gk + 1e-12)) + 1e-12)

        # max over tokens, mean over heads
        loss = loss.max(dim=-1).values.mean(dim=1)
        losses.append(loss.mean())

    losses = torch.stack(losses)
    return losses.mean() if reduction == "mean" else losses.sum()


class film(torch.nn.Module):
    def __init__(self, h_dim=768, in_dim=1):
        super().__init__()
        self.condition_to_scale = nn.Sequential(
            nn.Linear(in_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, h_dim),
            )

        self.condition_to_shift = nn.Sequential(
            nn.Linear(in_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, h_dim),
            )

    def forward(self, x, t):
        scale = self.condition_to_scale(t[..., None])  # (..., C)
        shift = self.condition_to_shift(t[..., None])  # (..., C)
        if x.dim() == 3:
            scale = rearrange(scale, 'b c -> b 1 c')
            shift = rearrange(shift, 'b c -> b 1 c')
        elif x.dim() == 5:
            scale = rearrange(scale, 't c -> 1 t 1 1 c')
            shift = rearrange(shift, 't c -> 1 t 1 1 c')
        
        y = x * scale + shift
        return x + y



import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# small RMSNorm used if q/k normalization requested (matches Qwen idea)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # cos, sin: (batch, seq, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # hidden_states: (batch, n_kv_heads, seqlen, head_dim)
    if n_rep == 1:
        return hidden_states
    b, n_kv, slen, hdim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, slen, hdim)
    return hidden_states.reshape(b, n_kv * n_rep, slen, hdim)

def g_k(p, k=1):
    """
    Compute g_k(p) = x_(k) * (1 - x_(k) + x_(k+1))
    where x_(k) and x_(k+1) are k-th and (k+1)-th largest elements per row.
    """
    sorted_p, _ = torch.sort(p, dim=-1, descending=True)
    x_k = sorted_p[..., k - 1]
    x_k1 = sorted_p[..., k] if k < p.size(-1) else torch.zeros_like(x_k)
    return x_k * (1 - x_k + x_k1)


def jasmin_loss(attn_maps, k=0, reduction="mean"):
    """
    Compute JaSMin regularization loss.

    attn_maps: [B, H, N, N] tensor or list of them.
    k=0 -> Eq.(8) from paper
    k>0 -> Eq.(9)
    """
    if isinstance(attn_maps, torch.Tensor):
        attn_maps = [attn_maps]

    losses = []
    for P in attn_maps:
        P = torch.clamp(P, min=1e-12, max=1.0)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
        g1 = g_k(P, k=1)
        if k == 0:
            loss = torch.log(g1 + 1e-12)
        else:
            gk = g_k(P, k=k)
            loss = torch.log((g1 / (gk + 1e-12)) + 1e-12)

        # max over tokens, mean over heads
        loss = loss.max(dim=-1).values.mean(dim=1)
        losses.append(loss.mean())

    losses = torch.stack(losses)
    return losses.mean() if reduction == "mean" else losses.sum()


