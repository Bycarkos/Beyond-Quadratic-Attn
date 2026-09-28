import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from transformers.utils.generic import ModelOutput

import models.layers as layers


_EPS = 1e-4


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


class CenterNorm(nn.Module):
    """Same CenterNorm used by ODE_ViT."""

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


def _to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def _positive(raw: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Positive reparameterization used by SSOG."""
    return F.softplus(raw) + _EPS + floor


class SSOGAttention(nn.Module):
    """
    PyTorch Separable Sum-of-Gaussians attention.

    This mirrors the public SSOG implementation while keeping the ODE: forward(x) -> (out, attentions).

    Normal execution never constructs an N x N patch attention matrix. Set
    capture_attention=True only for visualization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_h: int,
        grid_w: int,
        num_atoms: int = 4,
        lookat: bool = True,
        max_offset: float = 4.0,
        max_sigma_log_offset: float = 1.0,
        cold_init: bool = True,
        steer_mu: bool = True,
        steer_sigma: bool = True,
        steer_lambda: bool = True,
        raw_mu_delta_init: Optional[float] = None,
        raw_sigma_delta_init: Optional[float] = None,
        raw_lambda_gate_init: Optional[float] = None,
        sigma_floor: float = 0.25,
        proj_drop: float = 0.0,
        has_cls_token: bool = True,
        capture_attention: bool = False,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_atoms = num_atoms
        self.lookat = lookat
        self.max_offset = max_offset
        self.max_sigma_log_offset = max_sigma_log_offset
        self.cold_init = cold_init
        self.steer_mu = steer_mu
        self.steer_sigma = steer_sigma
        self.steer_lambda = steer_lambda
        self.sigma_floor = sigma_floor
        self.has_cls_token = has_cls_token
        self.capture_attention = capture_attention

        # Query-source displacement grid never changes after construction;
        # precomputing it once avoids rebuilding it on every call.
        ys = torch.arange(grid_h, dtype=torch.float32)
        xs = torch.arange(grid_w, dtype=torch.float32)
        self.register_buffer("_dy_grid", ys[:, None] - ys[None, :], persistent=False)
        self.register_buffer("_dx_grid", xs[:, None] - xs[None, :], persistent=False)

        # Optional externally supplied (ay, ax, lam) for fixed SSOG, set by
        # SSOG_ViT.forward() so the ODE solver's repeated NFE evaluations
        # reuse one kernel instead of recomputing it at every step. None (the
        # default, and the only state for every other caller such as
        # DiscreteSSOGViT) means "compute it fresh inside forward() below".
        self._fixed_kernel_override = None

        # No Q or K projections in SSOG.
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learned per-head Gaussian atoms: (mu_y, mu_x), (sigma_y, sigma_x).
        self.mu = nn.Parameter(
            torch.empty(num_heads, num_atoms, 2).normal_(mean=0.0, std=0.5)
        )
        self.raw_sigma = nn.Parameter(
            torch.full((num_heads, num_atoms, 2), -0.5)
        )
        self.log_lambda = nn.Parameter(torch.zeros(num_heads, num_atoms))
        self.raw_temperature = nn.Parameter(torch.tensor(-1.0))

        if lookat:
            steering_dim = num_heads * num_atoms

            self.mu_delta = nn.Linear(dim, steering_dim * 2, bias=True)
            self.sigma_delta = nn.Linear(dim, steering_dim * 2, bias=True)
            self.lambda_gate = nn.Linear(dim, steering_dim, bias=True)

            # Content steering starts at exactly zero.
            nn.init.zeros_(self.mu_delta.weight)
            nn.init.zeros_(self.mu_delta.bias)
            nn.init.zeros_(self.sigma_delta.weight)
            nn.init.zeros_(self.sigma_delta.bias)
            nn.init.zeros_(self.lambda_gate.weight)
            nn.init.zeros_(self.lambda_gate.bias)

            # Cold-start the scalar gates, as in the reference implementation.
            raw_gate_init = -8.0 if cold_init else -4.0
            self.raw_mu_delta_scale = nn.Parameter(
                torch.tensor(
                    raw_gate_init
                    if raw_mu_delta_init is None
                    else raw_mu_delta_init
                )
            )
            self.raw_sigma_delta_scale = nn.Parameter(
                torch.tensor(
                    raw_gate_init
                    if raw_sigma_delta_init is None
                    else raw_sigma_delta_init
                )
            )
            self.raw_lambda_gate_scale = nn.Parameter(
                torch.tensor(
                    raw_gate_init
                    if raw_lambda_gate_init is None
                    else raw_lambda_gate_init
                )
            )

    def _displacements(self, x: torch.Tensor):
        # Precomputed once at construction (`_dy_grid`/`_dx_grid`); `.to()` is
        # a no-op once dtype/device already match.
        dy = self._dy_grid.to(dtype=x.dtype, device=x.device)
        dx = self._dx_grid.to(dtype=x.dtype, device=x.device)
        return dy, dx

    @staticmethod
    def _log_kernel(
        d: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return (
            -0.5
            * torch.log(
                torch.tensor(2.0 * torch.pi, dtype=d.dtype, device=d.device)
            )
            - torch.log(sigma)
            - (d - mu).square() / (2.0 * sigma.square())
        )

    def _compute_fixed_kernel(self, dy: torch.Tensor, dx: torch.Tensor):
        """
        Build fixed SSOG's (ay, ax, lam). These depend only on learned
        parameters, not on the input tokens, so they are identical at every
        ODE function evaluation within one forward pass. SSOG_ViT computes
        this once per forward pass (see set_fixed_kernel_override) instead of
        at every NFE step; any other caller (e.g. DiscreteSSOGViT) still gets
        it computed fresh inside forward() below.
        """
        sigma = _positive(self.raw_sigma, floor=self.sigma_floor)
        lam = torch.softmax(self.log_lambda, dim=-1)
        temperature = F.softplus(self.raw_temperature) + 0.5

        sy, sx = sigma[..., 0], sigma[..., 1]
        mu_y, mu_x = self.mu[..., 0], self.mu[..., 1]

        log_ay = self._log_kernel(
            dy[None, None, :, :],
            mu_y[:, :, None, None],
            sy[:, :, None, None],
        )
        log_ax = self._log_kernel(
            dx[None, None, :, :],
            mu_x[:, :, None, None],
            sx[:, :, None, None],
        )

        # [H, R, query, source]
        ay = torch.softmax(log_ay / temperature, dim=-1)
        ax = torch.softmax(log_ax / temperature, dim=-1)
        return ay, ax, lam

    def set_fixed_kernel_override(self, kernel):
        """Supply a precomputed (ay, ax, lam) for every forward() call until
        cleared. Used by SSOG_ViT to share one fixed-SSOG kernel across an
        entire ODE trajectory instead of recomputing it at each NFE step."""
        self._fixed_kernel_override = kernel

    def clear_fixed_kernel_override(self):
        self._fixed_kernel_override = None

    def _fixed_apply(
        self,
        v: torch.Tensor,
        ay: torch.Tensor,
        ax: torch.Tensor,
        lam: torch.Tensor,
    ):
        """Purely geometric SSOG: the field does not depend on x."""
        # Two separable 1-D filtering passes, no N x N matrix.
        y = torch.einsum("hrij,bjwhd->biwhdr", ay, v)
        y = torch.einsum("hrwk,bikhdr->biwhdr", ax, y)
        y = torch.einsum("hr,biwhdr->biwhd", lam, y)

        dense_attention = None
        if self.capture_attention:
            dense_attention = self._dense_fixed_attention(
                ay=ay,
                ax=ax,
                lam=lam,
                batch_size=v.shape[0],
            )

        return y, dense_attention

    def _steered_apply(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        log_lambda: torch.Tensor,
        temperature: torch.Tensor,
    ):
        """
        Content-steered SSOG (LookAt).

        Content changes only Gaussian geometry (mu, sigma, lambda); it never
        forms pairwise query-key scores.
        """
        b = x.shape[0]
        gh, gw = self.grid_h, self.grid_w
        h, r = self.num_heads, self.num_atoms

        x_grid = x.reshape(b, gh, gw, self.dim)

        # mu: bounded center displacement.
        mu_scale = _positive(self.raw_mu_delta_scale)
        if self.steer_mu:
            dmu = self.mu_delta(x_grid).reshape(b, gh, gw, h, r, 2)
            mu_y = (
                mu[None, None, None, :, :, 0]
                + mu_scale * self.max_offset * torch.tanh(dmu[..., 0])
            )
            mu_x = (
                mu[None, None, None, :, :, 1]
                + mu_scale * self.max_offset * torch.tanh(dmu[..., 1])
            )
        else:
            mu_y = mu[None, None, None, :, :, 0].expand(b, gh, gw, -1, -1)
            mu_x = mu[None, None, None, :, :, 1].expand(b, gh, gw, -1, -1)

        # sigma: bounded multiplicative width steering in log-space. The
        # log-space excursion is capped by max_sigma_log_offset (mirroring
        # how max_offset caps mu's displacement), and the result is floored
        # at sigma_floor -- without this floor, an unbounded sigma_scale
        # could otherwise steer sigma arbitrarily close to zero for some
        # queries, silently bypassing the same floor that section 10.1's
        # ablation found necessary for the un-steered case.
        sigma_scale = _positive(self.raw_sigma_delta_scale)
        if self.steer_sigma:
            dsig = self.sigma_delta(x_grid).reshape(b, gh, gw, h, r, 2)
            sy = torch.clamp_min(
                sigma[None, None, None, :, :, 0]
                * torch.exp(
                    sigma_scale * self.max_sigma_log_offset * torch.tanh(dsig[..., 0])
                ),
                self.sigma_floor,
            )
            sx = torch.clamp_min(
                sigma[None, None, None, :, :, 1]
                * torch.exp(
                    sigma_scale * self.max_sigma_log_offset * torch.tanh(dsig[..., 1])
                ),
                self.sigma_floor,
            )
        else:
            sy = sigma[None, None, None, :, :, 0].expand(b, gh, gw, -1, -1)
            sx = sigma[None, None, None, :, :, 1].expand(b, gh, gw, -1, -1)

        # [B, Gh, Gw, H, R, source_axis]
        log_ay = self._log_kernel(
            dy[None, :, None, None, None, :],
            mu_y[..., None],
            sy[..., None],
        )
        log_ax = self._log_kernel(
            dx[None, None, :, None, None, :],
            mu_x[..., None],
            sx[..., None],
        )

        ay = torch.softmax(log_ay / temperature, dim=-1)
        ax = torch.softmax(log_ax / temperature, dim=-1)

        # Same sequential factorized application used by the SSOG reference.
        y = torch.einsum("biwhrj,bjwhd->biwhdr", ay, v)
        y = torch.einsum("biwhrk,bikhdr->biwhdr", ax, y)

        # lambda: choose which Gaussian atoms matter for each query patch.
        lambda_scale = _positive(self.raw_lambda_gate_scale)
        if self.steer_lambda:
            gate = self.lambda_gate(x_grid).reshape(b, gh, gw, h, r)
            lam_q = torch.softmax(
                log_lambda[None, None, None, :, :]
                + lambda_scale * torch.tanh(gate),
                dim=-1,
            )
        else:
            lam_q = torch.softmax(log_lambda, dim=-1)[None, None, None].expand(
                b, gh, gw, -1, -1
            )

        y = torch.einsum("biwhr,biwhdr->biwhd", lam_q, y)

        dense_attention = None
        if self.capture_attention:
            dense_attention = self._dense_steered_attention(
                ay=ay,
                ax=ax,
                lam_q=lam_q,
            )

        return y, dense_attention

    def _add_cls_to_attention(
        self,
        patch_attention: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_cls_token:
            return patch_attention

        b, h, npatch, _ = patch_attention.shape
        full = patch_attention.new_zeros(b, h, npatch + 1, npatch + 1)
        full[:, :, 1:, 1:] = patch_attention

        # Matches the CLS update in forward(): uniform global patch pooling.
        full[:, :, 0, 1:] = 1.0 / npatch
        return full

    def _dense_fixed_attention(
        self,
        ay: torch.Tensor,
        ax: torch.Tensor,
        lam: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Dense matrix only for analysis/JaSMin."""
        a = torch.einsum("hrij,hrwk,hr->hriwjk", ay, ax, lam)
        a = a.sum(dim=1)
        a = a.reshape(
            self.num_heads,
            self.grid_h * self.grid_w,
            self.grid_h * self.grid_w,
        )
        a = a.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self._add_cls_to_attention(a)

    def _dense_steered_attention(
        self,
        ay: torch.Tensor,
        ax: torch.Tensor,
        lam_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dense effective matrix matching the sequential LookAt implementation.
        Only used when capture_attention=True.
        """
        # Effective coefficient from source (j,k) to query (i,w):
        # sum_r lambda(i,w,r) * ax(i,w,r,k) * ay(i,k,r,j)
        a = torch.einsum(
            "biwhr,biwhrk,bikhrj->biwhjk",
            lam_q,
            ax,
            ay,
        )
        a = a.permute(0, 3, 1, 2, 4, 5).contiguous()
        a = a.reshape(
            a.shape[0],
            self.num_heads,
            self.grid_h * self.grid_w,
            self.grid_h * self.grid_w,
        )
        return self._add_cls_to_attention(a)

    def forward(self, x: torch.Tensor):
        b, n, _ = x.shape
        num_patches = self.grid_h * self.grid_w
        expected_tokens = num_patches + int(self.has_cls_token)

        if n != expected_tokens:
            raise ValueError(
                f"SSOG expected {expected_tokens} tokens "
                f"({self.grid_h}x{self.grid_w} patches"
                f"{' + CLS' if self.has_cls_token else ''}), got {n}."
            )

        v = self.v_proj(x)

        if self.has_cls_token:
            x_patch = x[:, 1:, :]
            v_patch = v[:, 1:, :]
        else:
            x_patch = x
            v_patch = v

        v_grid = v_patch.reshape(
            b,
            self.grid_h,
            self.grid_w,
            self.num_heads,
            self.head_dim,
        )

        dy, dx = self._displacements(x)

        if self.lookat:
            sigma = _positive(self.raw_sigma, floor=self.sigma_floor)
            temperature = F.softplus(self.raw_temperature) + 0.5
            patch_y, attentions = self._steered_apply(
                x=x_patch,
                v=v_grid,
                dy=dy,
                dx=dx,
                mu=self.mu,
                sigma=sigma,
                log_lambda=self.log_lambda,
                temperature=temperature,
            )
        else:
            if self._fixed_kernel_override is not None:
                ay, ax, lam = self._fixed_kernel_override
            else:
                ay, ax, lam = self._compute_fixed_kernel(dy, dx)
            patch_y, attentions = self._fixed_apply(v=v_grid, ay=ay, ax=ax, lam=lam)

        patch_y = patch_y.reshape(b, num_patches, self.dim)

        if self.has_cls_token:
            # CLS has no 2-D coordinate. Give it a minimal global aggregation
            # path while leaving patch-to-patch mixing purely Gaussian.
            cls_y = v_patch.mean(dim=1, keepdim=True)
            y = torch.cat([cls_y, patch_y], dim=1)
        else:
            y = patch_y

        y = self.out_proj(y)
        y = self.proj_drop(y)

        return y, attentions


class Lie_Trotter(nn.Module):
    """
    Parallel sublayers used as an ODE derivative:

        dX/dt = gate_attn(t) * SSOG(X) + gate_mlp(t) * MLP(X)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_h: int,
        grid_w: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        num_atoms: int = 4,
        lookat: bool = True,
        max_offset: float = 4.0,
        max_sigma_log_offset: float = 1.0,
        cold_init: bool = True,
        steer_mu: bool = True,
        steer_sigma: bool = True,
        steer_lambda: bool = True,
        raw_mu_delta_init: Optional[float] = None,
        raw_sigma_delta_init: Optional[float] = None,
        raw_lambda_gate_init: Optional[float] = None,
        sigma_floor: float = 0.25,
        capture_attention: bool = False,
        has_cls_token: bool = True,
    ):
        super().__init__()

        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.dim = dim

        self.attn = SSOGAttention(
            dim=dim,
            num_heads=num_heads,
            grid_h=grid_h,
            grid_w=grid_w,
            num_atoms=num_atoms,
            lookat=lookat,
            max_offset=max_offset,
            max_sigma_log_offset=max_sigma_log_offset,
            cold_init=cold_init,
            steer_mu=steer_mu,
            steer_sigma=steer_sigma,
            steer_lambda=steer_lambda,
            raw_mu_delta_init=raw_mu_delta_init,
            raw_sigma_delta_init=raw_sigma_delta_init,
            raw_lambda_gate_init=raw_lambda_gate_init,
            sigma_floor=sigma_floor,
            proj_drop=proj_drop,
            has_cls_token=has_cls_token,
            capture_attention=capture_attention,
        )

        self.mlp = layers.MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop=mlp_drop,
        )

        self.nfe = 0
        self.time_film = layers.TimeFiLM(dim)
        self.attentions = None

        # Optional precomputed TimeFiLM(t_grid) table, set by SSOG_ViT.forward()
        # only when solver="euler" on a fixed grid (see set_time_film_table).
        # None (the default, and the state for any other caller/solver) means
        # "compute time_film(t) fresh below", exactly as before.
        self._time_film_table = None

    def set_time_film_table(self, table: torch.Tensor):
        """Row k must equal self.time_film(t_grid[k], ...); used only for a
        fixed-step Euler trajectory where nfe - 1 indexes the grid point each
        call is evaluated at."""
        self._time_film_table = table

    def clear_time_film_table(self):
        self._time_film_table = None

    def forward(self, x: torch.Tensor, t, *args, **kwargs) -> torch.Tensor:
        self.nfe += 1

        table = self._time_film_table
        step = self.nfe - 1
        if table is not None and step < table.shape[0]:
            params = table[step : step + 1]
        else:
            params = self.time_film(t, dtype=x.dtype, device=x.device)
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = params.chunk(6, dim=-1)

        x_attn = self.norm_attn(x)
        x_attn = (
            x_attn * (1 + scale_attn[:, None, :])
            + shift_attn[:, None, :]
        )

        x_mlp = self.norm_mlp(x)
        x_mlp = (
            x_mlp * (1 + scale_mlp[:, None, :])
            + shift_mlp[:, None, :]
        )

        g, self.attentions = self.attn(x_attn)
        f = self.mlp(x_mlp)

        gate_attn = torch.sigmoid(gate_attn)[:, None, :]
        gate_mlp = torch.sigmoid(gate_mlp)[:, None, :]

        return gate_attn * g + gate_mlp * f


class Integrator(nn.Module):
    """ODE function dX/dt = F(t, X)."""

    def __init__(self, equation: nn.Module):
        super().__init__()
        self.block = equation
        self.attention_trajectory = []

    def integrate(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dx = self.block(x, t)

        if self.block.attentions is not None:
            self.attention_trajectory.append(self.block.attentions)

        return dx

    def forward(self, t, tokens: torch.Tensor) -> torch.Tensor:
        return self.integrate(t, tokens)


class SSOG_ViT(nn.Module):
    """
    ODE-ViT with SSOG attention.

    Constructor remains close to ODE_ViT for config/training compatibility.

    lookat=False -> fixed SSOG
    lookat=True  -> content-steered SSOG
    """

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
        return_error_upper_bound: bool = False,
        num_atoms: int = 4,
        lookat: bool = True,
        max_offset: float = 4.0,
        max_sigma_log_offset: float = 1.0,
        cold_init: bool = True,
        steer_mu: bool = True,
        steer_sigma: bool = True,
        steer_lambda: bool = True,
        raw_mu_delta_init: Optional[float] = None,
        raw_sigma_delta_init: Optional[float] = None,
        raw_lambda_gate_init: Optional[float] = None,
        sigma_floor: float = 0.25,
        capture_attention: bool = False,
        has_cls_token: bool = True,
        use_custom_euler_loop: bool = False,
    ):
        super().__init__()

        img_h, img_w = _to_2tuple(img_size)
        patch_h, patch_w = _to_2tuple(patch_size)

        if img_h % patch_h != 0 or img_w % patch_w != 0:
            raise ValueError("img_size must be divisible by patch_size for SSOG.")

        self.grid_h = img_h // patch_h
        self.grid_w = img_w // patch_w

        self.equation = Lie_Trotter(
            dim=dim,
            num_heads=num_heads,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            num_atoms=num_atoms,
            lookat=lookat,
            max_offset=max_offset,
            max_sigma_log_offset=max_sigma_log_offset,
            cold_init=cold_init,
            steer_mu=steer_mu,
            steer_sigma=steer_sigma,
            steer_lambda=steer_lambda,
            raw_mu_delta_init=raw_mu_delta_init,
            raw_sigma_delta_init=raw_sigma_delta_init,
            raw_lambda_gate_init=raw_lambda_gate_init,
            sigma_floor=sigma_floor,
            capture_attention=capture_attention,
            has_cls_token=has_cls_token,
        )

        self.integrator = Integrator(equation=self.equation)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

        self.return_error_upper_bound = return_error_upper_bound
        self.solver = solver
        self.embed_dim = dim
        self.num_eval_steps = num_eval_steps
        self.regularize = regularize
        self.lookat = lookat
        self.has_cls_token = has_cls_token
        self.capture_attention = capture_attention
        self.use_custom_euler_loop = use_custom_euler_loop

        self.patch_embeddings = layers.ViTEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            dim=dim,
        )

        # SSOG is already globally Lipschitz by construction, so retain the
        # standard PyTorch initialization instead of rescaling every linear
        # map through an expensive spectral decomposition.
        self._post_init_weights()

    def _post_init_weights(self):
        """Initialize embeddings and residual outputs without spectral rescaling."""
        with torch.no_grad():
            if hasattr(self.patch_embeddings, "cls_token"):
                nn.init.trunc_normal_(self.patch_embeddings.cls_token, std=0.02)

            if hasattr(self.patch_embeddings, "pos_embed"):
                nn.init.trunc_normal_(self.patch_embeddings.pos_embed, std=0.02)

            classifier_head = self.classifier[-1]
            nn.init.trunc_normal_(classifier_head.weight, std=0.02)
            if classifier_head.bias is not None:
                nn.init.zeros_(classifier_head.bias)

            self.equation.attn.out_proj.weight.mul_(0.1)

            if hasattr(self.equation.mlp, "fc2"):
                self.equation.mlp.fc2.weight.mul_(0.1)

    @staticmethod
    def g_k(p, k=1):
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
        Existing JaSMin regularizer, available only when dense SSOG attention
        is explicitly captured.
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

    def _euler_rollout(self, y0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        """
        Hand-rolled fixed-step explicit Euler integration.

        Equivalent to torchdiffeq.odeint(..., method="euler") for exactly how
        this model calls it: t_grid supplies both the integration grid and
        the output query points, with the library defaults this repo never
        overrides (identity grid_constructor since no step_size/
        grid_constructor option is passed, perturb=Perturb.NONE since no
        options={"perturb": True} is passed, interp="linear"). Verified
        against torchdiffeq's FixedGridODESolver.integrate() and Euler.
        _step_func(): each step calls the vector field once at t0 and adds
        dt * f0, and _linear_interp(t0, t1, y0, y1, t) returns y1 exactly
        (no interpolation arithmetic) whenever the query time equals t1,
        which is always the case here since every query point is itself a
        grid point. So this loop simply stores the state after every step.

        Only valid for solver="euler" on this fixed t_grid; SSOG_ViT.forward
        only takes this path when self.solver == "euler".
        """
        solution = [y0]
        y = y0
        for k in range(t_grid.shape[0] - 1):
            t0 = t_grid[k]
            dt = t_grid[k + 1] - t0
            f0 = self.integrator(t0, y)
            y = y + dt * f0
            solution.append(y)
        return torch.stack(solution, dim=0)

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
        tokens = self.patch_embeddings(pixel_values)

        if t_grid is None:
            t_grid = torch.linspace(
                0.0,
                1.0,
                steps=self.num_eval_steps,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            t_grid = t_grid.to(device=tokens.device, dtype=tokens.dtype)

        self.integrator.attention_trajectory = []

        # Fixed SSOG's kernel doesn't depend on the tokens, so it is
        # identical at every one of the solver's NFE evaluations: compute it
        # once here instead of inside each of the repeated calls to the same
        # module that odeint makes over one trajectory.
        if not self.lookat:
            attn = self.equation.attn
            attn.set_fixed_kernel_override(
                attn._compute_fixed_kernel(*attn._displacements(tokens))
            )

        # TimeFiLM(t) only depends on the scalar time, not on the tokens, and
        # explicit fixed-step Euler evaluates the vector field at exactly the
        # points of t_grid (verified against torchdiffeq's FixedGridODESolver:
        # identity grid_constructor, perturb=Perturb.NONE, linear interp never
        # calls func again). So batch it over the whole grid once instead of
        # once per NFE step. Guarded to solver="euler" specifically, since
        # other solvers evaluate at additional, data-dependent sub-step times
        # this table would not cover; Lie_Trotter.forward falls back to
        # computing it fresh whenever no table is set.
        if self.solver == "euler":
            self.equation.set_time_film_table(
                self.equation.time_film.forward_grid(
                    t_grid, dtype=tokens.dtype, device=tokens.device
                )
            )

        try:
            if self.use_custom_euler_loop and self.solver == "euler":
                out = self._euler_rollout(tokens, t_grid)
            else:
                out = odeint(
                    self.integrator,
                    tokens,
                    t=t_grid,
                    method=self.solver,
                    rtol=5e-1,
                    atol=5e-1,
                )
        finally:
            if not self.lookat:
                self.equation.attn.clear_fixed_kernel_override()
            if self.solver == "euler":
                self.equation.clear_time_film_table()

        if self.has_cls_token:
            pooled = out[-1][:, 0]
        else:
            pooled = out[-1].mean(dim=1)

        output = {
            "hidden_states": out,
            "last_hidden_state": out[-1],
            "logits": self.classifier(pooled),
            "nfe": copy.copy(self.equation.nfe),
        }

        if (
            output_attentions_trajectory
            and len(self.integrator.attention_trajectory) > 0
        ):
            output["attentions_trajectory"] = torch.stack(
                self.integrator.attention_trajectory,
                dim=0,
            )

        if output_attentions:
            output["attentions"] = self.equation.attentions

        regularize = kwargs.get("regularize", self.regularize)

        if regularize:
            output["lyapunov_loss"] = None

            if len(self.integrator.attention_trajectory) > 0:
                output["jasmin_loss"] = self.jasmin_loss(
                    attn_maps=self.integrator.attention_trajectory,
                    k=kwargs.get("k_jasmin", 4),
                )
            else:
                # Keeps existing loss code compatible while avoiding N x N
                # materialization by default.
                output["jasmin_loss"] = out.new_zeros(())

        # Do not reuse the old SDPA upper bound: SSOG has no Wq/Wk.
        if return_error_upper_bound or self.return_error_upper_bound:
            output["error_upper_bound"] = None

        if labels is not None:
            output["cross_entropy"] = F.cross_entropy(
                output["logits"],
                labels,
                reduction="mean",
            )

        self.equation.nfe = 0
        self.integrator.attention_trajectory = []

        if not return_dict:
            return tuple(v for v in output.values() if v is not None)

        return BaseModelOutput(**output)


# Alias for simpler imports.
SSOGViT = SSOG_ViT
