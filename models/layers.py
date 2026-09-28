import copy
import math
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MultiheadSelfAttention(nn.Module):
    """
    Thin wrapper around nn.MultiheadAttention to work with [B, N, D] shapes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=bias,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> attn_out same shape
        attn_out, attentions = self.mha(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        attn_out = self.proj_drop(attn_out)
        return (attn_out, attentions)


class SimpleGatedSDPAttention(nn.Module):
    """
    Simplified SDPA with optional gating (headwise or elementwise), optional q/k RMSNorm,
    RoPE-compatible (expects precomputed cos,sin), and simple past_key_value (concatenate).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        gate: str = "none",  # "none" | "headwise" | "elementwise"
        gate_shared: bool = False,  # if True, gating weights shared across heads
        gate_activation: str = "sigmoid",  # "sigmoid" or "silu"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.num_kv_heads = num_key_value_heads or num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(
            hidden_size, self.num_heads * self.head_dim, bias=qkv_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, hidden_size, bias=qkv_bias
        )

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            # we normalize per head-dim
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # gating
        assert gate in ("none", "headwise", "elementwise")
        self.gate_type = gate
        self.gate_shared = gate_shared
        self.gate_activation = gate_activation

        if self.gate_type != "none":
            # compute gate scores from the same input X (as in paper's default G1)
            # headwise: one scalar per head -> out dim = num_heads (or 1 if shared)
            # elementwise: vector per head -> out dim = num_heads * head_dim (or head_dim if shared)
            if self.gate_type == "headwise":
                out_dim = 1 if self.gate_shared else self.num_heads
            else:  # elementwise
                out_dim = (
                    self.head_dim
                    if self.gate_shared
                    else (self.num_heads * self.head_dim)
                )

            self.gate_proj = nn.Linear(hidden_size, out_dim, bias=False)

    def _compute_gate(self, x, attn_out):
        # x: (batch, seq, hidden)
        # attn_out: (batch, seq, num_heads, head_dim)
        if self.gate_type == "none":
            return attn_out

        gate_logits = self.gate_proj(x)  # (batch, seq, out_dim)
        if self.gate_activation == "sigmoid":
            gate_scores = torch.sigmoid(gate_logits)
        elif self.gate_activation == "silu":
            gate_scores = F.silu(gate_logits)
        elif self.gate_activation == "tanh":
            gate_scores = torch.tanh(gate_logits)
        else:
            gate_scores = gate_logits  # fallback

        b, seq = gate_scores.shape[0], gate_scores.shape[1]

        if self.gate_type == "headwise":
            # make shape (b, seq, num_heads, 1)
            if self.gate_shared:
                # gate_scores: (b, seq, 1)
                gate = gate_scores.unsqueeze(-1).unsqueeze(-1)  # -> (b,seq,1,1)
                gate = gate.expand(b, seq, self.num_heads, 1)
            else:
                # gate_scores: (b, seq, num_heads)
                gate = gate_scores.unsqueeze(-1)  # -> (b,seq,num_heads,1)
        else:  # elementwise
            if self.gate_shared:
                # gate_scores: (b, seq, head_dim)
                gate = gate_scores.unsqueeze(2)  # -> (b,seq,1,head_dim)
                gate = gate.expand(b, seq, self.num_heads, self.head_dim)
            else:
                # gate_scores: (b, seq, num_heads * head_dim)
                gate = gate_scores.view(b, seq, self.num_heads, self.head_dim)

        return attn_out * gate  # broadcast multiply

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin) or None
        output_attentions: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        hidden_states: (batch, seq, hidden)
        attention_mask: (batch, 1, seq, seq_kv) or None (should be additive mask where masked positions are -inf/very negative)
        past_key_value: optional tuple (past_k, past_v)
            each of shape (batch, num_kv_heads, past_seq, head_dim)
        position_embeddings: (cos, sin) each (batch, seq, head_dim) if using RoPE
        """
        bsz, seq_len, _ = hidden_states.size()

        # projections
        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )  # (b, seq, heads, hdim)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        # optional q/k RMSNorm
        if self.use_qk_norm:
            # apply per-head RMSNorm (RMSNorm expects last dim to be head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)

        # apply RoPE if provided (cos,sin)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(
                q, k, cos, sin, unsqueeze_dim=2
            )  # adjusts shapes for (b,heads,seq,hdim) if needed

        # handle past key/value (simple concat)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # expected past_k: (b, num_kv_heads, past_seq, head_dim)
            # current k/v are (b, seq, num_kv_heads, head_dim) -> transpose reorder
            k = k.transpose(1, 2)  # -> (b, num_kv_heads, seq, head_dim)
            v = v.transpose(1, 2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            # bring back to (b, seq_all, num_kv_heads, head_dim) for consistent shapes below
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # repeat KV to match number of heads if num_kv_heads < num_heads
        k = repeat_kv(k.transpose(1, 2), self.num_heads // self.num_kv_heads).transpose(
            1, 2
        )  # -> (b, seq_all, heads, hdim)
        v = repeat_kv(v.transpose(1, 2), self.num_heads // self.num_kv_heads).transpose(
            1, 2
        )

        # move to (b, heads, seq, hdim) for matmul
        q = q.transpose(1, 2)  # (b, heads, seq_q, h)
        k = k.transpose(1, 2)  # (b, heads, seq_k, h)
        v = v.transpose(1, 2)  # (b, heads, seq_k, h)

        # scaled dot product
        attn_scores = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (b, heads, seq_q, seq_k)

        if attention_mask is not None:
            # attention_mask should be additive (very negative where masked)
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            attn_scores.dtype
        )
        attn_out = torch.matmul(attn_probs, v)  # (b, heads, seq_q, h)

        # bring to (b, seq, heads, h)
        attn_out = attn_out.transpose(1, 2)  # (b, seq, heads, h)

        # apply gating (paper: gating after SDPA output is most effective)
        gated = self._compute_gate(hidden_states, attn_out)  # (b, seq, heads, h)

        # concat heads and final projection
        gated = gated.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        output = self.out_proj(gated)

        attn_weights = attn_probs if output_attentions else None

        return output, attn_weights


class CenterNorm(nn.Module):
    r"""CenterNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = normalized_shape / (normalized_shape - 1.0)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = self.scale * (x - u)
        x = self.weight * x + self.bias
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Small sinusoidal time embedding followed by an MLP."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] or [B, 1]
        if t.ndim == 2:
            t = t[:, 0]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device))
            * torch.arange(half, device=device)
            / max(half - 1, 1)
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class ViTEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed
        return x


class TimeFiLM(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * dim),
        )

    def forward(self, t, dtype, device):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)

        t = t.to(device=device, dtype=dtype).reshape(1, 1)
        params = self.net(t)
        return params

    def forward_grid(self, t_grid, dtype, device):
        """
        Batch this module over an entire fixed time grid at once. Row k of
        the result equals forward(t_grid[k], ...): nn.Linear applies the same
        weights independently to each row, so batching changes nothing but
        how many kernel launches it costs to evaluate every grid point.
        """
        t = t_grid.to(device=device, dtype=dtype).reshape(-1, 1)
        return self.net(t)

class PatchEmbed(nn.Module):
    """Image to patch tokens using a Conv2d projection."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TransformerVectorField(nn.Module):
    """
    One-layer transformer block used as an ODE vector field dH/dt = f_theta(H, t).

    The time embedding is added to all tokens, making the vector field time-dependent.
    The CLS derivative is image-dependent after attention to patch tokens.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, t: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # H: [B, N+1, D]
        if len(t.shape) == 0:
            t = t.repeat(H.shape[0])
        te = self.time_embed(t).unsqueeze(1)
        X = H + te
        A, _ = self.attn(
            self.norm1(X), self.norm1(X), self.norm1(X), need_weights=False
        )
        X = X + A
        X = X + self.mlp(self.norm2(X))
        # Return a derivative, not a residual state.
        return X - H
