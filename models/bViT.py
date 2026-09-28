import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils.generic import ModelOutput

import models.layers as layers


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[torch.FloatTensor] = None
    attentions_trajectory: Optional[torch.FloatTensor] = None
    error_upper_bound: Optional[float] = None
    logits: Optional[torch.FloatTensor] = None
    cross_entropy: Optional[torch.FloatTensor] = None
    lyapunov_loss: Optional[torch.FloatTensor] = None
    jasmin_loss: Optional[torch.FloatTensor] = None
    nfe: Optional[int] = None


class RMSNorm(nn.Module):
    """
    RMSNorm used by bViT.

    The paper uses a pre-norm Transformer with RMS normalization:
        z = x + MHSA(RMSNorm(x))
        x = z + FFN(RMSNorm(z))
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class MultiHeadSelfAttention(nn.Module):
    """
    Standard dense multi-head self-attention used by bViT.

    When output_attentions=False, PyTorch SDPA is used.
    When output_attentions=True, the N x N attention matrix is explicitly
    constructed so it can be returned for analysis.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
    ):
        b, n, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            b,
            n,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        attentions = None

        if output_attentions:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attentions = torch.softmax(scores, dim=-1)
            attentions = F.dropout(
                attentions,
                p=self.attn_drop,
                training=self.training,
            )
            out = torch.matmul(attentions, v)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
            )

        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out, attentions


class RecurrentTransformerBlock(nn.Module):
    """
    The single Transformer block reused at every bViT step.

    Paper equations:
        z^t = x^{t-1} + MHSA(RMSNorm(x^{t-1}))
        x^t = z^t       + FFN(RMSNorm(z^t))

    Importantly, this module is instantiated exactly once by BViT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()

        self.norm_attn = RMSNorm(dim)
        self.norm_mlp = RMSNorm(dim)

        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.mlp = layers.MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop=mlp_drop,
        )

        self.nfe = 0
        self.attentions = None

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        self.nfe += 1

        g, self.attentions = self.attn(
            self.norm_attn(x),
            output_attentions=output_attentions,
        )
        z = x + g

        f = self.mlp(self.norm_mlp(z))
        x = z + f

        return x


class BViT(nn.Module):
    """
    Single-block recurrent Vision Transformer (bViT).

    The ODE-ViT interface is retained where useful, but the computation is
    discrete:

        x^0 = ViTEmbed(image)
        x^t = F(x^{t-1}),  t = 1, ..., T

    where F is ONE shared Transformer block.

    Optional variants from the paper:
        use_time_embeddings=True:
            x^{t+1} = F(x^t + TE^t)

        num_registers=4:
            add four learned register tokens.

    Notes on compatibility
    ----------------------
    * num_eval_steps is accepted as the default recurrent depth so existing
      ODE-ViT configs can instantiate this model.
    * num_steps, when supplied, takes precedence over num_eval_steps.
    * solver and t_grid are accepted for interface compatibility but are not
      used: bViT is not an ODE solver.
    * nfe is used as "number of shared-block evaluations", i.e. T.
    """

    def __init__(
        self,
        img_size,
        patch_size,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_eval_steps: int = 12,
        num_steps: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        gated_attention: bool = False,
        solver: Optional[str] = None,
        regularize: bool = False,
        num_classes: int = 10,
        return_error_upper_bound: bool = False,
        use_time_embeddings: bool = False,
        num_registers: int = 0,
        capture_attention: bool = False,
        **kwargs,
    ):
        super().__init__()

        if num_steps is None:
            num_steps = num_eval_steps

        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        if num_registers < 0:
            raise ValueError("num_registers must be >= 0.")

        # Kept only so a common model configuration can instantiate BViT.
        # They do not alter the paper's architecture.
        self.gated_attention = gated_attention
        self.solver = solver

        self.embed_dim = dim
        self.num_steps = num_steps
        self.num_eval_steps = num_steps
        self.regularize = regularize
        self.return_error_upper_bound = return_error_upper_bound
        self.use_time_embeddings = use_time_embeddings
        self.num_registers = num_registers
        self.capture_attention = capture_attention

        self.patch_embeddings = layers.ViTEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            dim=dim,
        )

        # Exactly ONE block is instantiated and reused for all recurrent steps.
        self.block = RecurrentTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
        )

        if use_time_embeddings:
            self.time_embeddings = nn.Parameter(
                torch.zeros(num_steps, dim)
            )
            nn.init.trunc_normal_(self.time_embeddings, std=0.02)
        else:
            self.time_embeddings = None

        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, dim)
            )
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        # The paper normalizes the final CLS token using RMSNorm before the
        # linear classifier.
        self.classifier = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_classes),
        )

        self._post_init_weights()

    def _post_init_weights(self):
        """
        Conservative initialization for components introduced in this module.

        ViTEmbed and layers.MLP keep the initialization defined in your
        existing codebase.
        """
        with torch.no_grad():
            classifier_head = self.classifier[-1]
            nn.init.trunc_normal_(classifier_head.weight, std=0.02)

            if classifier_head.bias is not None:
                nn.init.zeros_(classifier_head.bias)

    def _add_register_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.register_tokens is None:
            return tokens

        if tokens.shape[1] < 1:
            raise ValueError(
                "bViT register tokens require a CLS token at index 0."
            )

        b = tokens.shape[0]
        registers = self.register_tokens.expand(b, -1, -1)

        # ViTEmbed has already added CLS and positional embeddings.
        # Registers are inserted immediately after CLS.
        return torch.cat(
            [
                tokens[:, :1],
                registers,
                tokens[:, 1:],
            ],
            dim=1,
        )

    @staticmethod
    def g_k(p, k=1):
        """
        Kept for compatibility with the JaSMin regularization code used by
        your ODE-ViT models. It is not part of the original bViT objective.
        """
        sorted_p, _ = torch.sort(p, dim=-1, descending=True)
        x_k = sorted_p[..., k - 1]
        x_k1 = (
            sorted_p[..., k]
            if k < p.size(-1)
            else torch.zeros_like(x_k)
        )
        return x_k * (1 - x_k + x_k1)

    def jasmin_loss(self, attn_maps, k=0, reduction="mean"):
        """
        Optional compatibility regularizer.

        The original bViT paper does not train with JaSMin. Leave
        regularize=False for a faithful baseline.
        """
        if isinstance(attn_maps, torch.Tensor):
            attn_maps = [attn_maps]

        eps = 1e-8
        losses = []

        for idx, p in enumerate(attn_maps):
            p = torch.clamp(p, min=eps, max=1.0)
            p = p / (p.sum(dim=-1, keepdim=True) + eps)

            current_k = len(attn_maps) - idx + 1
            current_k = min(max(2, current_k), p.size(-1) - 1)

            g1 = self.g_k(p, k=1)
            gk = self.g_k(p, k=current_k)

            loss = torch.log(g1 + eps) - torch.log(gk + eps)
            loss = loss.mean(dim=-1).mean(dim=-1)
            losses.append(loss.mean())

        losses = torch.stack(losses)
        return losses.mean() if reduction == "mean" else losses.sum()

    def forward(
        self,
        pixel_values: torch.Tensor,
        t_grid: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
        output_attentions_trajectory: bool = False,
        return_error_upper_bound: bool = False,
        max_norm_bounding: int = 12,
        return_dict: bool = True,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del t_grid
        del max_norm_bounding

        tokens = self.patch_embeddings(pixel_values)
        tokens = self._add_register_tokens(tokens)

        # Allows temporary depth overrides for recurrence/early-exit studies:
        #
        #   model(x, num_steps=8)
        #
        # without changing the trained model.
        num_steps = kwargs.get("num_steps", self.num_steps)

        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        if self.time_embeddings is not None and num_steps > self.num_steps:
            raise ValueError(
                f"Requested {num_steps} recurrent steps, but only "
                f"{self.num_steps} learned time embeddings exist."
            )

        regularize = kwargs.get("regularize", self.regularize)

        hidden_states = [tokens]
        attention_trajectory = []
        last_attention = None

        for step in range(num_steps):
            # Eq. (9) in the paper for bViT-TE:
            #     x^{t+1} = F(x^t + TE^t)
            if self.time_embeddings is not None:
                tokens = (
                    tokens
                    + self.time_embeddings[step][None, None, :]
                )

            # Avoid constructing N x N matrices unless they are actually
            # requested. For output_attentions=True only the final step needs
            # to expose its attention map.
            capture_this_step = (
                self.capture_attention
                or output_attentions_trajectory
                or regularize
                or (output_attentions and step == num_steps - 1)
            )

            tokens = self.block(
                tokens,
                output_attentions=capture_this_step,
            )

            hidden_states.append(tokens)

            if capture_this_step:
                last_attention = self.block.attentions

                if output_attentions_trajectory or regularize:
                    attention_trajectory.append(last_attention)

        hidden_states = torch.stack(hidden_states, dim=0)

        # bViT uses the final CLS token.
        pooled = tokens[:, 0]
        logits = self.classifier(pooled)

        output = {
            "hidden_states": hidden_states,
            "last_hidden_state": tokens,
            "logits": logits,
            # Compatibility with ODE-ViT: here NFE means the number of
            # evaluations of the recurrent shared block.
            "nfe": copy.copy(self.block.nfe),
        }

        if output_attentions:
            output["attentions"] = last_attention

        if (
            output_attentions_trajectory
            and len(attention_trajectory) > 0
        ):
            output["attentions_trajectory"] = torch.stack(
                attention_trajectory,
                dim=0,
            )

        if regularize:
            output["lyapunov_loss"] = None

            if len(attention_trajectory) > 0:
                output["jasmin_loss"] = self.jasmin_loss(
                    attn_maps=attention_trajectory,
                    k=kwargs.get("k_jasmin", 4),
                )
            else:
                output["jasmin_loss"] = logits.new_zeros(())

        # The ODE truncation/error bound does not apply to a discrete bViT.
        if return_error_upper_bound or self.return_error_upper_bound:
            output["error_upper_bound"] = None

        if labels is not None:
            output["cross_entropy"] = F.cross_entropy(
                logits,
                labels,
                reduction="mean",
            )

        self.block.nfe = 0
        self.block.attentions = None

        if not return_dict:
            return tuple(
                value
                for value in output.values()
                if value is not None
            )

        return BaseModelOutput(**output)


# -------------------------------------------------------------------------
# Paper configurations
# -------------------------------------------------------------------------

def bvit_s(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    **kwargs,
):
    """
    bViT-S from Table 1:
        dim=384, heads=6, recurrent steps=12
    """
    return BViT(
        img_size=img_size,
        patch_size=patch_size,
        dim=384,
        num_heads=6,
        num_steps=12,
        num_classes=num_classes,
        **kwargs,
    )


def bvit_b(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    **kwargs,
):
    """
    bViT-B from Table 1:
        dim=768, heads=12, recurrent steps=12
    """
    return BViT(
        img_size=img_size,
        patch_size=patch_size,
        dim=768,
        num_heads=12,
        num_steps=12,
        num_classes=num_classes,
        **kwargs,
    )


def bvit_l(
    **kwargs,
):
    """
    bViT-L from Table 1:
        dim=1024, heads=16, recurrent steps=24
    """
    return BViT(
        **kwargs,
    )


# Alias matching your existing naming style.
bViT = BViT
