"""Discrete ViT variants using SSOG token mixing and no ODE solver."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.layers as layers
from models.SSOGViT import BaseModelOutput, SSOGAttention


class DiscreteSSOGBlock(nn.Module):
    """Original ViT pre-norm/GELU block with SSOG replacing only MHSA."""

    def __init__(self, dim, num_heads, grid_size, mlp_ratio, num_atoms,
                 lookat, sigma_floor, max_offset, cold_init, proj_drop, mlp_drop,
                 capture_attention, num_special_tokens=0):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SSOGAttention(
            dim=dim, num_heads=num_heads, grid_h=grid_size, grid_w=grid_size,
            num_atoms=num_atoms, lookat=lookat, sigma_floor=sigma_floor,
            max_offset=max_offset, cold_init=cold_init, proj_drop=proj_drop,
            has_cls_token=num_special_tokens > 0,
            capture_attention=capture_attention,
        )
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = layers.MLP(dim, int(dim * mlp_ratio), drop=mlp_drop)
        self.mlp.act = nn.GELU()

    def _mix(self, normalized):
        if self.num_special_tokens != 2:
            return self.attn(normalized)
        # SSOG represents one coordinate-free global token. For DeiT, evaluate
        # the patch operator once and give CLS/DIST the same global update;
        # their residual streams and classifier heads remain distinct.
        primary = torch.cat([normalized[:, :1], normalized[:, 2:]], dim=1)
        primary_update, attention = self.attn(primary)
        special_update = primary_update[:, :1].expand(-1, 2, -1)
        update = torch.cat([special_update, primary_update[:, 1:]], dim=1)
        return update, attention

    def forward(self, tokens):
        mixed, attention = self._mix(self.norm_attn(tokens))
        tokens = tokens + mixed
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens, attention


class DiscreteSSOG_ViT(nn.Module):
    """Discrete SSOG transformer with independent or recurrent blocks.

    The default is the established patch-only discrete model. ``recurrent``
    shares one block across ``depth`` function evaluations for bViT matching.
    ``distilled`` adds DeiT-style CLS/DIST tokens and two classifier heads.
    """

    def __init__(self, img_size, patch_size, dim, num_heads, depth=12,
                 mlp_ratio=4.0, num_classes=100, num_atoms=4, lookat=False,
                 sigma_floor=0.25, max_offset=7.0, cold_init=True,
                 proj_drop=0.0, mlp_drop=0.0, capture_attention=False,
                 recurrent=False, distilled=False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if not isinstance(img_size, int) or not isinstance(patch_size, int):
            raise ValueError("Discrete SSOG requires scalar image and patch sizes.")
        if patch_size <= 0 or img_size <= 0 or img_size % patch_size:
            raise ValueError("img_size must be positive and divisible by patch_size.")
        self.depth = depth
        self.embed_dim = dim
        self.recurrent = recurrent
        self.distilled = distilled
        self.num_special_tokens = 2 if distilled else 0
        self.patch_embeddings = DiscretePatchEmbeddings(img_size, patch_size, dim)
        if distilled:
            self.special_tokens = nn.Parameter(torch.zeros(1, 2, dim))
            self.special_pos_embed = nn.Parameter(torch.zeros(1, 2, dim))
        block_count = 1 if recurrent else depth
        self.blocks = nn.ModuleList([
            DiscreteSSOGBlock(
                dim, num_heads, img_size // patch_size, mlp_ratio, num_atoms,
                lookat, sigma_floor, max_offset, cold_init, proj_drop, mlp_drop,
                capture_attention, self.num_special_tokens,
            ) for _ in range(block_count)
        ])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.classifier = nn.Linear(dim, num_classes)
        self.distillation_head = nn.Linear(dim, num_classes) if distilled else None
        self.apply(self._init_weights)
        with torch.no_grad():
            nn.init.trunc_normal_(self.patch_embeddings.pos_embed, std=0.02)
            if distilled:
                nn.init.trunc_normal_(self.special_tokens, std=0.02)
                nn.init.trunc_normal_(self.special_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            if self.distillation_head is not None:
                nn.init.trunc_normal_(self.distillation_head.weight, std=0.02)
            for block in self.blocks:
                block.attn.out_proj.weight.mul_(0.1)
                block.mlp.fc2.weight.mul_(0.1)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            with torch.no_grad():
                norm = torch.linalg.svdvals(module.weight)[0].clamp_min(1e-6)
                module.weight.div_(norm)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, pixel_values, labels=None, output_attentions=True,
                output_attentions_trajectory=False, output_hidden_states=False,
                return_dict=True, **kwargs):
        del kwargs
        tokens = self.patch_embeddings(pixel_values)
        if self.distilled:
            special = self.special_tokens.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([special + self.special_pos_embed, tokens], dim=1)
        states = [tokens] if output_hidden_states else None
        attention_maps = []
        attention = None
        blocks = [self.blocks[0]] * self.depth if self.recurrent else self.blocks

        # When recurrent=True, every one of the `depth` calls above reuses
        # the same block, so fixed SSOG's kernel (which depends only on
        # learned parameters, not on the tokens) is identical at every step.
        # Compute it once here instead of inside each repeated call,
        # mirroring SSOG_ViT's analogous optimization for its ODE solver's
        # NFE steps. Steered (lookat=True) kernels depend on the tokens and
        # must still be recomputed at every step, so this is skipped then.
        cached_kernel_attn = None
        if self.recurrent and not blocks[0].attn.lookat:
            cached_kernel_attn = blocks[0].attn
            cached_kernel_attn.set_fixed_kernel_override(
                cached_kernel_attn._compute_fixed_kernel(
                    *cached_kernel_attn._displacements(tokens)
                )
            )

        try:
            for block in blocks:
                tokens, attention = block(tokens)
                if states is not None:
                    states.append(tokens)
                if output_attentions_trajectory and attention is not None:
                    attention_maps.append(attention)
        finally:
            if cached_kernel_attn is not None:
                cached_kernel_attn.clear_fixed_kernel_override()

        tokens = self.final_norm(tokens)
        if self.distilled:
            logits = (self.classifier(tokens[:, 0])
                      + self.distillation_head(tokens[:, 1])) / 2
        else:
            logits = self.classifier(tokens.mean(dim=1))
        output = BaseModelOutput(
            last_hidden_state=tokens,
            hidden_states=torch.stack(states) if states is not None else None,
            logits=logits,
            attentions=attention if output_attentions else None,
            attentions_trajectory=torch.stack(attention_maps) if attention_maps else None,
            cross_entropy=F.cross_entropy(logits, labels) if labels is not None else None,
            nfe=self.depth if self.recurrent else None,
        )
        return output if return_dict else output.to_tuple()


DiscreteSSOGViT = DiscreteSSOG_ViT


class DiscretePatchEmbeddings(nn.Module):
    """Convolutional patch embedding with learned patch positions."""

    def __init__(self, img_size, patch_size, dim):
        super().__init__()
        self.patch_embed = layers.PatchEmbed(img_size, patch_size, 3, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim))

    def forward(self, images):
        return self.patch_embed(images) + self.pos_embed
