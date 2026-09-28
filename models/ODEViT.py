import copy
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdyn.models as dynmodels
import torchdyn.nn as dnn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchdiffeq import odeint, odeint_adjoint
from transformers.utils.generic import ModelOutput

import models.layers as layers
import models.utils as utils


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions_trajectory: Optional[tuple[torch.FloatTensor, ...]] = None
    error_upper_bound: Optional[float] = None
    logits: Optional[torch.FloatTensor] = None
    cross_entropy: Optional[torch.FloatTensor] = None
    lyapunov_loss: Optional[torch.FloatTensor] = None
    jasmin_loss: Optional[torch.FloatTensor] = None
    nfe: Optional[int] = None

    ## For Vision Encoder decoder models
    encoder_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None


class CenterNorm(nn.Module):
    r"""CenterNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape))
        self.scale = normalized_shape / (normalized_shape - 1.0)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = self.scale * (x - u)
        x = self.weight[None, :] * x + self.bias[None, :]
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, z_dim: int = 0, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim + z_dim, hidden_dim, bias=False)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, dim + z_dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


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
        capture_attention: bool = True,
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
        self.capture_attention = capture_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> attn_out same shape
        attn_out, attentions = self.mha(
            x,
            x,
            x,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        attn_out = self.proj_drop(attn_out)
        return (attn_out, attentions)


# Define the Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Lie_Trotter(nn.Module):
    """
    Implements the parallel sublayers: return F(x) + G(x,X) as a *derivative* (no residual add here).
    LayerNorms are applied before the sublayers (pre-norm).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        gated_attention: bool = True,
        capture_attention: bool = True,
    ):
        super().__init__()
        self.norm_attn = CenterNorm(dim)
        self.norm_mlp = CenterNorm(dim)
        self.dim = dim
        self.capture_attention = capture_attention

        self._set_ivp = None

        if gated_attention:
            self.attn = layers.SimpleGatedSDPAttention(
                hidden_size=dim,
                num_heads=num_heads,
                num_key_value_heads=num_heads,
                qkv_bias=False,
                use_qk_norm=False,
                gate="headwise",
                gate_shared=False,
                gate_activation="sigmoid",
            )

        else:
            self.attn = MultiheadSelfAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                capture_attention=capture_attention,
            )

        self.mlp = layers.MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=mlp_drop)

        self.nfe = 0
        self.data_control = dnn.DataControl()
        self.data_control.forward = lambda x: torch.cat(
            [x, self.data_control._control], -1
        ).to(x)

        self.time_film = layers.TimeFiLM(dim)

    @property
    def set_ivp(self):
        return self._set_ivp

    @set_ivp.setter
    def set_ivp(self, x: torch.Tensor):
        self._set_ivp = x

    def _prepare_control(self, x: torch.Tensor):
        if hasattr(self, "data_control"):
            self.data_control._control = x

    def forward(self, x: torch.Tensor, t, *args, **kwargs) -> torch.Tensor:

        self.nfe += 1

        params = self.time_film(t, dtype=x.dtype, device=x.device)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        
        x_attn = self.norm_attn(x)
        x_attn = x_attn * (1 + scale_attn[:, None, :]) + shift_attn[:, None, :]
        
        x_mlp = self.norm_mlp(x)
        x_mlp = x_mlp * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        
        if isinstance(self.attn, layers.SimpleGatedSDPAttention):
            g, self.attentions = self.attn(
                x_attn, output_attentions=self.capture_attention
            )
        else:
            g, self.attentions = self.attn(x_attn)
        f = self.mlp(x_mlp)
        
        gate_attn = torch.sigmoid(gate_attn)[:, None, :]
        gate_mlp = torch.sigmoid(gate_mlp)[:, None, :]
        
        return gate_attn * g + gate_mlp * f


class Integrator(nn.Module):
    """
    ODE function f(t, x) that returns dx/dt = F(x) + G(x).
    Optionally time-scales the drift if integrating over [0, 1] but wanting to emulate D discrete layers.
    """

    def __init__(
        self,
        equation: nn.Module,
    ):
        super().__init__()
        self.block = equation

    def integrate(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        t = t.to(x.device)

        dx = self.block(x, t)

        # record attentions for this time step
        if not hasattr(self, "attention_trajectory"):
            self.attention_trajectory = []
        self.attention_trajectory.append(self.block.attentions)

        return dx

    def forward(self, t, tokens: torch.Tensor) -> torch.Tensor:
        return self.integrate(t, tokens)


class ODE_ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_eval_steps: int = 24,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        gated_attention: bool = False,
        solver: str = "euler",
        regularize: bool = False,
        num_classes: int = 10,
        return_error_upper_bound: bool = True,
        capture_attention: bool = True,
    ):

        super().__init__()


        ## Change the equation in order to change the ODE dynamics
        self.equation = Lie_Trotter(
            dim,
            num_heads,
            mlp_ratio,
            attn_drop,
            proj_drop,
            mlp_drop,
            gated_attention,
            capture_attention,
        )

        self.integrator = Integrator(
            equation=self.equation,
        )

        self.lyapunov_kappa = 1.0
        self.lyapunov_max_points = 6
        self.lyapunov_detach_states = True

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.return_error_upper_bound = return_error_upper_bound
        self.solver = solver
        self.embed_dim = dim
        self.num_eval_steps = num_eval_steps

        self.solver = solver

        self.regularize = regularize

        self.patch_embeddings = layers.ViTEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, dim=dim
        )

        self.apply(self._spectral_init)
        self._post_init_weights()

    def _spectral_init(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            with torch.no_grad():
                spectral_norm = torch.linalg.svdvals(m.weight)[0].clamp_min(1e-6)
                m.weight.div_(spectral_norm)

        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, CenterNorm, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if m.weight is not None:
                nn.init.ones_(m.weight)

    def _post_init_weights(self):
        """
        Keep ODE dynamics conservative at initialization while preserving the
        spectral initialization used for the vector field internals.
        """
        with torch.no_grad():
            nn.init.trunc_normal_(self.patch_embeddings.cls_token, std=0.02)
            nn.init.trunc_normal_(self.patch_embeddings.pos_embed, std=0.02)

            classifier_head = self.classifier[-1]
            nn.init.trunc_normal_(classifier_head.weight, std=0.02)
            if classifier_head.bias is not None:
                nn.init.zeros_(classifier_head.bias)

            if hasattr(self.equation.attn, "out_proj"):
                self.equation.attn.out_proj.weight.mul_(0.1)
            elif hasattr(self.equation.attn, "mha"):
                self.equation.attn.mha.out_proj.weight.mul_(0.1)

            if hasattr(self.equation.mlp, "fc2"):
                self.equation.mlp.fc2.weight.mul_(0.1)

    @staticmethod
    def g_k(p, k=1):
        """
        Compute g_k(p) = x_(k) * (1 - x_(k) + x_(k+1))
        where x_(k) and x_(k+1) are k-th and (k+1)-th largest elements per row.
        """
        sorted_p, _ = torch.sort(p, dim=-1, descending=True)
        x_k = sorted_p[..., k - 1]
        x_k1 = sorted_p[..., k] if k < p.size(-1) else torch.zeros_like(x_k)
        return x_k * (1 - x_k + x_k1)

    def jasmin_loss(self, attn_maps, k=0, reduction="mean"):
        """
        Compute JaSMin regularization loss.

        attn_maps: [B, H, N, N] tensor or list of them.
        k=0 -> Eq.(8) from paper
        k>0 -> Eq.(9)
        """
        if isinstance(attn_maps, torch.Tensor):
            attn_maps = [attn_maps]

        eps = 1e-8
        losses = []
        for idx, P in enumerate(attn_maps):
            P = torch.clamp(P, min=eps, max=1.0)
            P = P / (P.sum(dim=-1, keepdim=True) + eps)
            
            current_k = len(attn_maps) - idx + 1
            current_k = min(max(2, current_k), P.size(-1) - 1)

            g1 = self.g_k(P, k=1)
            gk = self.g_k(P, k=current_k)
            
            loss = torch.log(g1 + eps) - torch.log(gk + eps)

            loss = loss.mean(dim=-1).mean(dim=-1)
            losses.append(loss.mean())

        losses = torch.stack(losses)
        return losses.mean() if reduction == "mean" else losses.sum()

    def compute_lyapunov_path_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        logits_fn,
        t_grid: Optional[torch.Tensor] = None,
        kappa: float = 1.0,
        max_points: Optional[int] = 4,
        detach_states: bool = True,
    ) -> torch.Tensor:
        """
        Lyapunov path loss for classification.

        hidden_states:
            Tensor from odeint with shape [T, B, N, D].

        labels:
            Class labels with shape [B].

        logits_fn:
            Function mapping eta -> logits.
            Example:
                logits_fn = lambda eta: classifier(eta[:, 0])

        kappa:
            Desired exponential decay rate.

        max_points:
            Number of trajectory points used to compute the loss.
            Use a small number to save memory.

        detach_states:
            If True, does not backpropagate through the ODE solver trajectory.
            This is much cheaper and closer to the paper's local Lyapunov training idea.
        """

        assert hidden_states.dim() == 4, (
            "Expected hidden_states with shape [T, B, N, D]"
        )

        T = hidden_states.shape[0]

        if t_grid is None:
            t_grid = self._last_t_grid

        t_grid = t_grid.to(hidden_states.device)

        # Select a few trajectory points.
        # We skip index 0 because it is the initial condition.
        if max_points is None or max_points >= T - 1:
            time_indices = torch.arange(1, T, device=hidden_states.device)
        else:
            time_indices = (
                torch.linspace(1, T - 1, steps=max_points, device=hidden_states.device)
                .long()
                .unique()
            )

        lyapunov_terms = []

        for idx in time_indices:
            eta = hidden_states[idx]

            if detach_states:
                eta = eta.detach().requires_grad_(True)
            else:
                eta.requires_grad_(True)

            # V_y(eta) = classification loss at this hidden state.
            logits = logits_fn(eta)
            V = F.cross_entropy(logits, labels, label_smoothing=.1, reduction="none")  # [B]

            # grad_eta V_y(eta)
            grad_eta = torch.autograd.grad(
                V.sum(),
                eta,
                create_graph=True,
                retain_graph=True,
            )[0]

            # f_theta(eta, x, t)
            t = t_grid[idx]
            f_eta = self.equation(eta, t)

            # <grad V, f>
            dVdt = (grad_eta * f_eta).flatten(1).sum(dim=1)  # [B]

            violation = dVdt + kappa * V
            lyapunov_terms.append(F.relu(violation))

        return torch.cat(lyapunov_terms, dim=0).mean()

    @torch.no_grad()
    def compute_upper_bound_by_second_derivative(
        self, R: int = 12, L: float = 0.5
    ) -> float:
        # mha.in_proj_weight
        try:
            Wq, Wk, Wv = self.equation.attn.mha.in_proj_weight.reshape(
                3, self.embed_dim, self.embed_dim
            )
        except:
            Wq = self.equation.attn.q_proj.weight
            Wk = self.equation.attn.k_proj.weight
            Wv = self.equation.attn.v_proj.weight

        factor1 = R**2 * torch.norm(Wv, p=2)
        factor2 = R * torch.linalg.norm(Wk @ Wq.mT) + (Wk.shape[-1]) ** 0.5
        factor3 = (self.num_eval_steps**2) * (Wq.shape[-1] ** 0.5)
        supremum = (factor1 * factor2) / factor3

        bound = (math.e ** (L) - 1) / (2 * L * self.num_eval_steps) * supremum

        return bound

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
    ) -> torch.Tensor:
        """
        tokens: [B, N, D]
        returns: [B, N, D]
        """
        tokens = self.patch_embeddings(pixel_values)  # [B, N, D]

        # token_noise_std = torch.rand(1).item()
        # if self.training and token_noise_std > 0.9:
        #     noise_scale = tokens.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        #     tokens = tokens + token_noise_std * noise_scale * torch.randn_like(tokens)

        if t_grid is None:
            t_grid = torch.linspace(0.0, 1.0, steps=self.num_eval_steps)
        else:
            t_grid = t_grid

        self.equation._prepare_control(tokens)
        out = odeint(
            self.integrator, tokens, t=t_grid.to(tokens.device), method=self.solver, rtol=5e-1, atol=5e-1
        )

        output = {
            "hidden_states": out,
            "last_hidden_state": out[-1],
            "logits": self.classifier(out[-1][:, 0]),  # [B, N, D] -> [B, N_classes]
            "nfe": copy.copy(self.equation.nfe),
        }

        self._forward_hidden_states = out

        if output_attentions_trajectory:
            output["encoder_attentions"] = torch.stack(
                self.integrator.attention_trajectory, dim=0
            )
            self._encoder_attentions = output["encoder_attentions"]

        if output_attentions:
            output["attentions"] = self.equation.attentions
            self._attention_last_head = output["attentions"]

        regularize = kwargs.get("regularize", self.regularize)

        if regularize:
            if self.training:
                # output["lyapunov_loss"] = self.compute_lyapunov_path_loss(
                #     hidden_states=output["hidden_states"],
                #     labels=labels,
                #     logits_fn=lambda eta: self.classifier(eta[:, 0]),
                #     t_grid=t_grid,
                #     kappa=self.lyapunov_kappa,
                #     max_points=self.lyapunov_max_points,
                #     detach_states=self.lyapunov_detach_states,
                # )
                output["lyapunov_loss"] = None
            if (
                hasattr(self.integrator, "attention_trajectory")
                and len(self.integrator.attention_trajectory) > 0
            ):
                output["jasmin_loss"] = self.jasmin_loss(
                    attn_maps=self.integrator.attention_trajectory,
                    k=kwargs.get("k_jasmin", 4),
                )

        if self.return_error_upper_bound:
            upper_bound = self.compute_upper_bound_by_second_derivative(
                R=max_norm_bounding, L=0.5
            )
            output["error_upper_bound"] = upper_bound
            self._error_upper_bound = upper_bound

        if hasattr(self.integrator, "attention_trajectory"):
            del self.integrator.attention_trajectory  # free memory
            self.integrator.attention_trajectory = []  # reset for next forward

        self.equation.nfe = 0

        if hasattr(self.integrator, "attention_trajectory"):
            self.integrator.attention_trajectory = []

        if labels is not None:
            output["cross_entropy"] = F.cross_entropy(
                output["logits"], labels, reduction="mean"
            )  # [B, N, num_classes] -> [B, N] -> [B]

        if not return_dict:
            return tuple(v for v in output.values() if v is not None)

        else:
            return BaseModelOutput(**output)
