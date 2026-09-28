"""Lightning trainer for SSOG-ViT image classification."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from trainers.loss_trainer_classification import TrainerClassification


class TrainerSSOGClassification(TrainerClassification):
    """Classification trainer with SSOG diagnostics and qualitative tracking.

    All optimization and classification metric names are inherited from
    :class:`TrainerClassification`. The first four validation tensors are kept
    for the lifetime of the trainer, so W&B compares atom movement on exactly
    the same transformed images after every validation epoch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_visualization_samples: int = 4,
        visualization_every_n_epochs: int = 1,
        visualization_query: Optional[tuple[int, int]] = None,
        visualization_queries: Optional[Sequence[Sequence[int]]] = None,
        visualization_mean: Sequence[float] = (0.485, 0.456, 0.406),
        visualization_std: Sequence[float] = (0.229, 0.224, 0.225),
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            *args,
            **kwargs,
        )
        if num_visualization_samples < 0:
            raise ValueError("num_visualization_samples must be non-negative.")
        if visualization_every_n_epochs < 1:
            raise ValueError("visualization_every_n_epochs must be at least 1.")

        self.num_visualization_samples = num_visualization_samples
        self.visualization_every_n_epochs = visualization_every_n_epochs
        self.visualization_query = visualization_query
        if visualization_queries is not None:
            if not visualization_queries:
                raise ValueError("visualization_queries cannot be empty.")
            if any(len(query) != 2 for query in visualization_queries):
                raise ValueError(
                    "Every visualization query must contain exactly two coordinates."
                )
            self.visualization_queries = [
                tuple(int(coordinate) for coordinate in query)
                for query in visualization_queries
            ]
        else:
            self.visualization_queries = None
        self.visualization_mean = tuple(float(value) for value in visualization_mean)
        self.visualization_std = tuple(float(value) for value in visualization_std)

        # These are deliberately not registered buffers: validation images are
        # diagnostics, should remain on CPU, and must not enter checkpoints.
        self._fixed_validation_images: list[torch.Tensor] = []
        self._fixed_validation_labels: list[int] = []

    @property
    def ssog_attention(self):
        equation = getattr(self.model, "equation", None)
        attention = getattr(equation, "attn", None)
        required = ("mu", "raw_sigma", "log_lambda", "grid_h", "grid_w")
        if attention is None or not all(hasattr(attention, name) for name in required):
            raise TypeError(
                "TrainerSSOGClassification requires a model whose "
                "model.equation.attn is an SSOGAttention module."
            )
        return attention

    @staticmethod
    def _pixel_tensor(batch) -> torch.Tensor:
        inputs = batch["pixel_values"]
        if isinstance(inputs, dict):
            return inputs["pixel_values"]
        return inputs

    def _remember_validation_samples(self, batch):
        if len(self._fixed_validation_images) >= self.num_visualization_samples:
            return

        images = self._pixel_tensor(batch).detach().float().cpu()
        labels = batch.get("hard_labels", batch["labels"])
        labels = self._hard_labels(labels).detach().long().cpu()
        remaining = self.num_visualization_samples - len(
            self._fixed_validation_images
        )

        for image, label in zip(images[:remaining], labels[:remaining]):
            self._fixed_validation_images.append(image.clone())
            self._fixed_validation_labels.append(int(label.item()))

    def validation_step(self, batch, batch_idx):
        self._remember_validation_samples(batch)
        return super().validation_step(batch, batch_idx)

    @staticmethod
    def _positive(raw: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
        return F.softplus(raw) + 1.0e-4 + floor

    def _ssog_statistics(self):
        attention = self.ssog_attention
        sigma = self._positive(attention.raw_sigma, attention.sigma_floor)
        weights = torch.softmax(attention.log_lambda, dim=-1)
        entropy = -(
            weights * weights.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        effective_atoms = entropy.exp()
        temperature = F.softplus(attention.raw_temperature) + 0.5

        statistics = {
            "mu_abs_mean": attention.mu.abs().mean(),
            "mu_max": attention.mu.abs().max(),
            "sigma_mean": sigma.mean(),
            "sigma_min": sigma.min(),
            "sigma_max": sigma.max(),
            "lambda_entropy": entropy.mean(),
            "effective_atoms": effective_atoms.mean(),
            "lambda_max": weights.max(dim=-1).values.mean(),
            "temperature": temperature,
        }

        if getattr(attention, "lookat", False):
            zero = attention.raw_mu_delta_scale.new_zeros(())
            statistics.update(
                {
                    "mu_delta_scale": self._positive(
                        attention.raw_mu_delta_scale
                    )
                    if attention.steer_mu
                    else zero,
                    "sigma_delta_scale": self._positive(
                        attention.raw_sigma_delta_scale
                    )
                    if attention.steer_sigma
                    else zero,
                    "lambda_gate_scale": self._positive(
                        attention.raw_lambda_gate_scale
                    )
                    if attention.steer_lambda
                    else zero,
                }
            )
        return statistics

    def _log_ssog_statistics(self, prefix: str):
        for name, value in self._ssog_statistics().items():
            self.log(
                f"{prefix}/ssog_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_train_epoch_end(self):
        self._log_ssog_statistics("train")

    def _resolve_query(
        self,
        grid_h: int,
        grid_w: int,
        query: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        query = query or self.visualization_query or (grid_h // 2, grid_w // 2)
        if not (0 <= query[0] < grid_h and 0 <= query[1] < grid_w):
            raise ValueError(
                f"visualization_query={query} is outside the "
                f"SSOG grid {grid_h}x{grid_w}."
            )
        return query

    def _resolve_visualization_queries(
        self, grid_h: int, grid_w: int
    ) -> list[tuple[int, int]]:
        queries = self.visualization_queries
        if queries is None:
            queries = [
                (grid_h // 4, grid_w // 4),
                (grid_h // 4, 3 * grid_w // 4),
                (grid_h // 2, grid_w // 2),
                (3 * grid_h // 4, grid_w // 4),
                (3 * grid_h // 4, 3 * grid_w // 4),
            ]
        return [
            self._resolve_query(grid_h, grid_w, tuple(query))
            for query in queries
        ]

    def _attention_input(self, images: torch.Tensor) -> torch.Tensor:
        """Capture the SSOG input from the final numerical ODE evaluation."""
        attention = self.ssog_attention
        captured_inputs: list[torch.Tensor] = []

        def capture_input(_module, module_inputs):
            captured_inputs.append(module_inputs[0].detach())

        hook = attention.register_forward_pre_hook(capture_input)
        try:
            self.model(
                pixel_values=images,
                output_attentions=False,
                output_attentions_trajectory=False,
                regularize=False,
            )
        finally:
            hook.remove()

        if not captured_inputs:
            raise RuntimeError("SSOG attention was not evaluated during model forward.")
        return captured_inputs[-1]

    def _atom_fields(
        self,
        images: torch.Tensor,
        query: Optional[tuple[int, int]] = None,
        attention_input: Optional[torch.Tensor] = None,
    ):
        """Compute query-specific per-atom fields and mixture weights."""
        attention = self.ssog_attention
        grid_h, grid_w = attention.grid_h, attention.grid_w
        query_y, query_x = self._resolve_query(grid_h, grid_w, query)
        x = (
            attention_input
            if attention_input is not None
            else self._attention_input(images)
        )
        x_patch = x[:, 1:] if attention.has_cls_token else x
        x_grid = x_patch.reshape(images.shape[0], grid_h, grid_w, attention.dim)

        base_sigma = self._positive(
            attention.raw_sigma, attention.sigma_floor
        )
        source_y = torch.arange(
            grid_h, device=images.device, dtype=images.dtype
        )
        source_x = torch.arange(
            grid_w, device=images.device, dtype=images.dtype
        )
        displacement_y = query_y - source_y
        displacement_x = query_x - source_x
        temperature = F.softplus(attention.raw_temperature) + 0.5

        if attention.lookat:
            batch_size = images.shape[0]
            heads, atoms = attention.num_heads, attention.num_atoms
            query_features = x_grid[:, query_y, query_x]

            # Horizontal filtering uses geometry at the selected query. The
            # preceding vertical pass uses geometry at (query_y, source_x),
            # exactly as in SSOGAttention._steered_apply.
            row_features = x_grid[:, query_y, :]
            if attention.steer_mu:
                row_delta_mu = attention.mu_delta(row_features).reshape(
                    batch_size, grid_w, heads, atoms, 2
                )
                query_delta_mu = attention.mu_delta(query_features).reshape(
                    batch_size, heads, atoms, 2
                )
                mu_scale = self._positive(attention.raw_mu_delta_scale)
                mu_y = (
                    attention.mu[None, None, :, :, 0]
                    + mu_scale
                    * attention.max_offset
                    * torch.tanh(row_delta_mu[..., 0])
                )
                mu_x = (
                    attention.mu[None, :, :, 1]
                    + mu_scale
                    * attention.max_offset
                    * torch.tanh(query_delta_mu[..., 1])
                )
            else:
                mu_y = attention.mu[None, None, :, :, 0].expand(
                    batch_size, grid_w, -1, -1
                )
                mu_x = attention.mu[None, :, :, 1].expand(
                    batch_size, -1, -1
                )

            if attention.steer_sigma:
                row_delta_sigma = attention.sigma_delta(row_features).reshape(
                    batch_size, grid_w, heads, atoms, 2
                )
                query_delta_sigma = attention.sigma_delta(query_features).reshape(
                    batch_size, heads, atoms, 2
                )
                sigma_scale = self._positive(attention.raw_sigma_delta_scale)
                sigma_y = base_sigma[None, None, :, :, 0] * torch.exp(
                    sigma_scale * torch.tanh(row_delta_sigma[..., 0])
                )
                sigma_x = base_sigma[None, :, :, 1] * torch.exp(
                    sigma_scale * torch.tanh(query_delta_sigma[..., 1])
                )
            else:
                sigma_y = base_sigma[None, None, :, :, 0].expand(
                    batch_size, grid_w, -1, -1
                )
                sigma_x = base_sigma[None, :, :, 1].expand(
                    batch_size, -1, -1
                )

            if attention.steer_lambda:
                gates = attention.lambda_gate(query_features).reshape(
                    batch_size, heads, atoms
                )
                gate_scale = self._positive(attention.raw_lambda_gate_scale)
                weights = torch.softmax(
                    attention.log_lambda[None]
                    + gate_scale * torch.tanh(gates),
                    dim=-1,
                )
            else:
                weights = torch.softmax(
                    attention.log_lambda, dim=-1
                )[None].expand(batch_size, -1, -1)

            # log_y: [B, source_x, H, R, source_y]
            log_y = attention._log_kernel(
                displacement_y[None, None, None, None, :],
                mu_y[..., None],
                sigma_y[..., None],
            )
            # log_x: [B, H, R, source_x]
            log_x = attention._log_kernel(
                displacement_x[None, None, None, :],
                mu_x[..., None],
                sigma_x[..., None],
            )
            axis_y = torch.softmax(log_y / temperature, dim=-1)
            axis_x = torch.softmax(log_x / temperature, dim=-1)
            fields = axis_y.permute(0, 2, 3, 4, 1) * axis_x[..., None, :]
        else:
            weights = torch.softmax(attention.log_lambda, dim=-1)[None].expand(
                images.shape[0], -1, -1
            )
            log_y = attention._log_kernel(
                displacement_y[None, None, None, :],
                attention.mu[None, ..., 0, None],
                base_sigma[None, ..., 0, None],
            )
            log_x = attention._log_kernel(
                displacement_x[None, None, None, :],
                attention.mu[None, ..., 1, None],
                base_sigma[None, ..., 1, None],
            )
            axis_y = torch.softmax(log_y / temperature, dim=-1)
            axis_x = torch.softmax(log_x / temperature, dim=-1)
            fields = (axis_y[..., :, None] * axis_x[..., None, :]).expand(
                images.shape[0], -1, -1, -1, -1
            )
        return fields, weights, (query_y, query_x)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[1]
        mean = images.new_tensor(self.visualization_mean[:channels]).view(
            1, channels, 1, 1
        )
        std = images.new_tensor(self.visualization_std[:channels]).view(
            1, channels, 1, 1
        )
        return (images * std + mean).clamp(0, 1)

    @staticmethod
    def _normalize_heatmap(field: torch.Tensor) -> torch.Tensor:
        field = field - field.amin(dim=(-2, -1), keepdim=True)
        return field / field.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)

    def _build_atom_figures(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        device = next(self.model.parameters()).device
        images = torch.stack(self._fixed_validation_images).to(device)
        attention_input = self._attention_input(images)
        fields, weights, query = self._atom_fields(
            images,
            attention_input=attention_input,
        )
        images_to_plot = self._denormalize(images).detach().cpu()
        fields = fields.detach()
        weights = weights.detach()
        atoms = fields.shape[2]
        figures = []

        for sample_index, (image, label) in enumerate(
            zip(images_to_plot, self._fixed_validation_labels)
        ):
            # Average each atom over heads, then show the actual weighted mixture.
            atom_fields = fields[sample_index].mean(dim=0)
            mixture = (
                fields[sample_index]
                * weights[sample_index, :, :, None, None]
            ).sum(dim=1).mean(dim=0)
            panels = torch.cat([atom_fields, mixture[None]], dim=0)
            panels = self._normalize_heatmap(panels).cpu()

            figure, axes = plt.subplots(
                1, atoms + 2, figsize=(3.2 * (atoms + 2), 3.4), squeeze=False
            )
            rgb = image.permute(1, 2, 0).numpy()
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title(f"sample {sample_index}, class={label}")
            axes[0, 0].axis("off")

            for atom_index in range(atoms):
                axis = axes[0, atom_index + 1]
                heatmap = F.interpolate(
                    panels[atom_index][None, None],
                    size=image.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
                mean_weight = weights[sample_index, :, atom_index].mean().item()
                axis.imshow(rgb)
                axis.imshow(heatmap.numpy(), cmap="magma", alpha=0.60, vmin=0, vmax=1)
                axis.set_title(f"atom {atom_index}\nlambda={mean_weight:.3f}")
                axis.axis("off")

            mixture_axis = axes[0, -1]
            mixture_heatmap = F.interpolate(
                panels[-1][None, None],
                size=image.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            mixture_axis.imshow(rgb)
            mixture_axis.imshow(
                mixture_heatmap.numpy(), cmap="magma", alpha=0.60, vmin=0, vmax=1
            )
            mixture_axis.set_title(f"mixture\nquery={query}")
            mixture_axis.axis("off")
            figure.tight_layout()
            figures.append(figure)
        queries = self._resolve_visualization_queries(
            self.ssog_attention.grid_h,
            self.ssog_attention.grid_w,
        )
        mixtures = []
        for mixture_query in queries:
            query_fields, query_weights, _ = self._atom_fields(
                images,
                query=mixture_query,
                attention_input=attention_input,
            )
            mixture = (
                query_fields
                * query_weights[:, :, :, None, None]
            ).sum(dim=2).mean(dim=1)
            mixtures.append(mixture)
        mixtures = self._normalize_heatmap(torch.stack(mixtures, dim=1)).cpu()

        mixture_figures = []
        for sample_index, (image, label) in enumerate(
            zip(images_to_plot, self._fixed_validation_labels)
        ):
            figure, axes = plt.subplots(
                1,
                len(queries) + 1,
                figsize=(3.2 * (len(queries) + 1), 3.4),
                squeeze=False,
            )
            rgb = image.permute(1, 2, 0).numpy()
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title(f"sample {sample_index}, class={label}")
            axes[0, 0].axis("off")

            for query_index, mixture_query in enumerate(queries):
                axis = axes[0, query_index + 1]
                heatmap = F.interpolate(
                    mixtures[sample_index, query_index][None, None],
                    size=image.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
                axis.imshow(rgb)
                axis.imshow(
                    heatmap.numpy(),
                    cmap="magma",
                    alpha=0.60,
                    vmin=0,
                    vmax=1,
                )
                axis.set_title(f"mixture query={mixture_query}")
                axis.axis("off")

            figure.tight_layout()
            mixture_figures.append(figure)

        return figures, mixture_figures

    @torch.no_grad()
    def _log_atom_visualizations(self):
        if not self._fixed_validation_images or self.logger is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        atom_figures, mixture_figures = self._build_atom_figures()
        try:
            if hasattr(experiment, "log"):
                import wandb

                atom_images = [
                    wandb.Image(
                        figure,
                        caption=(
                            f"fixed validation sample {index}; epoch={self.current_epoch}"
                        ),
                    )
                    for index, figure in enumerate(atom_figures)
                ]
                mixture_images = [
                    wandb.Image(
                        figure,
                        caption=(
                            f"fixed validation sample {index}; epoch={self.current_epoch}"
                        ),
                    )
                    for index, figure in enumerate(mixture_figures)
                ]
                experiment.log(
                    {
                        "val/ssog_atoms": atom_images,
                        "val/ssog_mixtures_grid": mixture_images,
                    },
                    step=self.global_step,
                )
            elif hasattr(experiment, "add_figure"):
                for index, figure in enumerate(atom_figures):
                    experiment.add_figure(
                        f"val/ssog_atoms/sample_{index}",
                        figure,
                        global_step=self.global_step,
                    )
                for index, figure in enumerate(mixture_figures):
                    experiment.add_figure(
                        f"val/ssog_mixtures_grid/sample_{index}",
                        figure,
                        global_step=self.global_step,
                    )
        finally:
            import matplotlib.pyplot as plt

            for figure in atom_figures + mixture_figures:
                plt.close(figure)

    def on_validation_epoch_end(self):
        self._log_ssog_statistics("val")
        if self.current_epoch % self.visualization_every_n_epochs == 0:
            self._log_atom_visualizations()


# Short alias matching the model's public SSOGViT alias.
TrainerSSOG = TrainerSSOGClassification
