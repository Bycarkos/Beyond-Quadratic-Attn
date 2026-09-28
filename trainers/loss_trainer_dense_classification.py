"""Lightning trainers for dense-attention classification baselines (ODE-ViT
and plain ViT/bViT), both reusing :class:`TrainerClassification` from
trainers/loss_trainer_classification.py -- the same "classification" trainer
family the SSOG trainer is built on -- instead of the visualization-less
:class:`TrainerBViTClassification`.

Both subclasses mirror :class:`TrainerSSOGClassification`'s qualitative
tracking (fixed validation samples, periodic W&B image logging) but for real
attention weights instead of an analytic Gaussian-mixture field, since dense
models have no SSOG module to introspect. Shared scaffolding (fixed-sample
bookkeeping, denormalization, figure layout, W&B logging) lives in
:class:`_AttentionVisualizationClassification`; each subclass only has to
say how to collect a labeled sequence of ``[B, H, N, N]`` attention tensors
from its own model.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from trainers.loss_trainer_classification import TrainerClassification


class _AttentionVisualizationClassification(TrainerClassification):
    """Shared scaffolding for logging attention-map figures during validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_visualization_samples: int = 4,
        visualization_every_n_epochs: int = 1,
        visualization_heads: Optional[Sequence[int]] = None,
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
        self.visualization_heads = (
            tuple(int(h) for h in visualization_heads)
            if visualization_heads is not None
            else None
        )
        self.visualization_mean = tuple(float(value) for value in visualization_mean)
        self.visualization_std = tuple(float(value) for value in visualization_std)

        # Diagnostics only: kept off the checkpoint/optimizer state on purpose.
        self._fixed_validation_images: list[torch.Tensor] = []
        self._fixed_validation_labels: list[int] = []

    @staticmethod
    def _pixel_tensor(batch) -> torch.Tensor:
        inputs = batch["pixel_values"]
        return inputs["pixel_values"] if isinstance(inputs, dict) else inputs

    def _remember_validation_samples(self, batch):
        if len(self._fixed_validation_images) >= self.num_visualization_samples:
            return
        images = self._pixel_tensor(batch).detach().float().cpu()
        labels = batch.get("hard_labels", batch["labels"])
        labels = self._hard_labels(labels).detach().long().cpu()
        remaining = self.num_visualization_samples - len(self._fixed_validation_images)
        for image, label in zip(images[:remaining], labels[:remaining]):
            self._fixed_validation_images.append(image.clone())
            self._fixed_validation_labels.append(int(label.item()))

    def validation_step(self, batch, batch_idx):
        self._remember_validation_samples(batch)
        return super().validation_step(batch, batch_idx)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[1]
        mean = images.new_tensor(self.visualization_mean[:channels]).view(1, channels, 1, 1)
        std = images.new_tensor(self.visualization_std[:channels]).view(1, channels, 1, 1)
        return (images * std + mean).clamp(0, 1)

    @staticmethod
    def _cls_attention_grid(attentions: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """attentions: [B, H, N, N] -> CLS-to-patch weights reshaped to a grid.

        Assumes a single leading CLS token at index 0 and a square patch grid.
        """
        cls_to_patches = attentions[:, :, 0, 1:]  # [B, H, num_patches]
        num_patches = cls_to_patches.shape[-1]
        grid = int(round(num_patches**0.5))
        if grid * grid != num_patches:
            raise ValueError(f"Expected a square patch grid, got {num_patches} patches.")
        return cls_to_patches.reshape(*cls_to_patches.shape[:-1], grid, grid), grid, grid

    def _collect_attention(self, images: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Return an ordered list of (label, attentions[B, H, N, N]) pairs.

        The last entry is treated as the "final" step and gets per-head panels
        in addition to its mean-over-heads panel; earlier entries only get a
        mean-over-heads panel, mirroring the SSOG per-NFE-step summary strip.
        Subclasses must implement this against their own model's internals.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _build_attention_figures(self):
        import matplotlib.pyplot as plt

        images = torch.stack(self._fixed_validation_images).to(self.device)
        steps = self._collect_attention(images)
        if not steps:
            return []

        grids = [
            (label, self._cls_attention_grid(attn.detach().float().cpu())[0])
            for label, attn in steps
        ]
        final_label, final_grid = grids[-1]
        num_heads = final_grid.shape[1]
        heads = self.visualization_heads or tuple(range(min(4, num_heads)))

        display_images = self._denormalize(torch.stack(self._fixed_validation_images)).cpu()

        figures = []
        for sample_index, (image, label) in enumerate(
            zip(display_images, self._fixed_validation_labels)
        ):
            rgb = image.permute(1, 2, 0).numpy()
            n_panels = 1 + len(heads) + len(grids)
            figure, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2), squeeze=False)
            axes = axes[0]

            axes[0].imshow(rgb)
            axes[0].set_title(f"sample {sample_index}, class={label}")
            axes[0].axis("off")

            def _overlay(axis, field, title):
                heatmap = F.interpolate(
                    field[None, None], size=image.shape[-2:], mode="bicubic", align_corners=False
                )[0, 0]
                axis.imshow(rgb)
                axis.imshow(heatmap.numpy(), cmap="magma", alpha=0.60)
                axis.set_title(title)
                axis.axis("off")

            for panel_index, head in enumerate(heads, start=1):
                _overlay(
                    axes[panel_index],
                    final_grid[sample_index, head],
                    f"head {head}\n({final_label})",
                )

            mean_start = 1 + len(heads)
            for offset, (step_label, step_grid) in enumerate(grids):
                _overlay(
                    axes[mean_start + offset],
                    step_grid[sample_index].mean(dim=0),
                    f"mean over heads\n({step_label})",
                )

            figure.tight_layout()
            figures.append(figure)
        return figures

    @torch.no_grad()
    def _log_attention_visualizations(self):
        if not self._fixed_validation_images or self.logger is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        figures = self._build_attention_figures()
        if not figures:
            return
        try:
            if hasattr(experiment, "log"):
                import wandb

                images = [
                    wandb.Image(
                        figure,
                        caption=f"fixed validation sample {index}; epoch={self.current_epoch}",
                    )
                    for index, figure in enumerate(figures)
                ]
                experiment.log({"val/attention_maps": images}, step=self.global_step)
            elif hasattr(experiment, "add_figure"):
                for index, figure in enumerate(figures):
                    experiment.add_figure(
                        f"val/attention_maps/sample_{index}",
                        figure,
                        global_step=self.global_step,
                    )
        finally:
            import matplotlib.pyplot as plt

            for figure in figures:
                plt.close(figure)

    def on_validation_epoch_end(self):
        if self.current_epoch % self.visualization_every_n_epochs == 0:
            self._log_attention_visualizations()


class TrainerDenseAttentionClassification(_AttentionVisualizationClassification):
    """Dense-attention ODE-ViT baseline: visualizes the CLS-to-patch attention
    at a few points along the ODE trajectory (analogous to the SSOG per-NFE-step
    atom visualization).

    Requires the model to have been constructed with ``capture_attention=True``
    (also required for the ``jasmin_loss`` regularizer the base trainer's
    ``training_step`` always requests -- ``need_weights=False`` makes
    ``nn.MultiheadAttention`` return ``None`` weights, which crashes
    ``jasmin_loss`` on the first step). This trainer refuses to run otherwise,
    rather than silently logging nothing.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        attention = self._dense_attention(model)
        if not getattr(attention, "capture_attention", False):
            raise ValueError(
                "TrainerDenseAttentionClassification requires the model's "
                "attention module to have capture_attention=True (needed both "
                "for these visualizations and for the base trainer's "
                "jasmin_loss regularizer, which crashes on None attention "
                "weights otherwise)."
            )

    @staticmethod
    def _dense_attention(model: torch.nn.Module):
        equation = getattr(model, "equation", None)
        attention = getattr(equation, "attn", None)
        if attention is None or not hasattr(attention, "mha"):
            raise TypeError(
                "TrainerDenseAttentionClassification requires a model whose "
                "model.equation.attn wraps nn.MultiheadAttention (the dense "
                "ODE-ViT baseline), not an SSOGAttention module."
            )
        return attention

    @property
    def dense_attention(self):
        return self._dense_attention(self.model)

    def _collect_attention(self, images: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        outputs = self.model(
            pixel_values=images,
            output_attentions=True,
            output_attentions_trajectory=True,
            regularize=False,
        )
        trajectory = outputs["encoder_attentions"]  # [S, B, H, N, N]
        num_steps = trajectory.shape[0]
        step_indices = sorted({0, num_steps // 2, num_steps - 1} if num_steps > 1 else {0})
        return [
            (f"ODE step {step}/{num_steps - 1}", trajectory[step]) for step in step_indices
        ]


class TrainerDenseViTAttentionClassification(_AttentionVisualizationClassification):
    """Dense standard-attention ViT/bViT baseline (MatchedViT / MatchedBViT).

    torchvision's ``EncoderBlock.forward`` hardcodes ``need_weights=False``, so
    attention weights are never available through the normal forward pass.
    This trainer replays each block's pre-norm/attention/residual computation
    manually for the fixed validation images only (never during normal
    training/validation steps, so there is no added cost or precision
    difference to the reported metrics), requesting real weights from the
    underlying ``nn.MultiheadAttention`` module directly.
    """

    def _collect_attention(self, images: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        model = self.model
        tokens = model._embed(images)
        blocks = list(model.blocks)
        # MatchedBViT reuses a single block `num_steps` times; MatchedViT has
        # one independent block per layer. Either way, `model.blocks` already
        # exposes the right per-evaluation sequence except for bViT's reuse.
        num_steps = getattr(model, "num_steps", None)
        if num_steps is not None:
            blocks = blocks * num_steps

        attentions = []
        for block in blocks:
            normed = block.ln_1(tokens)
            attn_out, weights = block.self_attention(
                normed, normed, normed, need_weights=True, average_attn_weights=False
            )
            tokens = tokens + block.dropout(attn_out)
            tokens = tokens + block.mlp(block.ln_2(tokens))
            attentions.append(weights)

        num_layers = len(attentions)
        layer_indices = sorted({0, num_layers // 2, num_layers - 1} if num_layers > 1 else {0})
        return [
            (f"layer {index}/{num_layers - 1}", attentions[index]) for index in layer_indices
        ]
