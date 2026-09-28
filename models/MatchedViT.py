"""Matched dense-attention ViT and single-block recurrent bViT baselines.

Both models use the same torchvision Vision Transformer components and differ
only in whether the Transformer blocks have independent parameters or one block
is reused recurrently.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

from models.bViT import BaseModelOutput, RMSNorm


class _MatchedViTBase(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        num_heads: int,
        num_classes: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if proj_drop != mlp_drop:
            raise ValueError(
                "Matched torchvision baselines require proj_drop == mlp_drop."
            )
        self.embed_dim = dim
        self.vit = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            dropout=mlp_drop,
            attention_dropout=attn_drop,
            num_classes=num_classes,
            norm_layer=RMSNorm,
        )

    def _embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.vit._process_input(pixel_values)
        cls = self.vit.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.vit.encoder.pos_embedding
        return self.vit.encoder.dropout(tokens)

    def _output(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor],
        hidden_states,
        nfe: Optional[int],
        return_dict: bool,
    ):
        tokens = self.vit.encoder.ln(tokens)
        logits = self.vit.heads(tokens[:, 0])
        output = BaseModelOutput(
            last_hidden_state=tokens,
            hidden_states=(
                torch.stack(hidden_states, dim=0)
                if hidden_states is not None
                else None
            ),
            logits=logits,
            cross_entropy=(
                F.cross_entropy(logits, labels)
                if labels is not None
                else None
            ),
            nfe=nfe,
        )
        return output if return_dict else output.to_tuple()


class MatchedViT(_MatchedViTBase):
    """Standard ViT with independently parameterized Transformer blocks."""

    def __init__(self, depth: int = 12, **kwargs):
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        super().__init__(num_layers=depth, **kwargs)
        self.depth = depth

    @property
    def blocks(self):
        return self.vit.encoder.layers

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        del kwargs
        tokens = self._embed(pixel_values)
        hidden_states = [tokens] if output_hidden_states else None
        for block in self.blocks:
            tokens = block(tokens)
            if hidden_states is not None:
                hidden_states.append(tokens)
        return self._output(
            tokens=tokens,
            labels=labels,
            hidden_states=hidden_states,
            nfe=None,
            return_dict=return_dict,
        )


class MatchedBViT(_MatchedViTBase):
    """bViT with one standard Transformer block reused for exactly T steps."""

    def __init__(self, num_steps: int = 12, **kwargs):
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        super().__init__(num_layers=1, **kwargs)
        self.num_steps = num_steps
        self.num_eval_steps = num_steps

    @property
    def block(self):
        return self.vit.encoder.layers[0]

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        del kwargs
        steps = self.num_steps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be at least 1.")

        tokens = self._embed(pixel_values)
        hidden_states = [tokens] if output_hidden_states else None
        for _ in range(steps):
            tokens = self.block(tokens)
            if hidden_states is not None:
                hidden_states.append(tokens)
        return self._output(
            tokens=tokens,
            labels=labels,
            hidden_states=hidden_states,
            nfe=steps,
            return_dict=return_dict,
        )

