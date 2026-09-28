from typing import Optional

import lightning as L
import torch


class TrainerBViTClassification(L.LightningModule):
    """
    Lightning trainer for bViT classification.

    Differences from the ODE-ViT trainer:
        - No ODE-specific regularization or error-bound logic.
        - `nfe` is interpreted as the number of recurrent evaluations of the
          shared Transformer block.
        - JaSMin is optional and disabled by default because it is not part of
          the original bViT training objective.

    The trainer remains compatible with the BaseModelOutput returned by the
    codebase-style BViT implementation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        jasmin_weight: float = 0.0,
        k_jasmin: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.jasmin_weight = jasmin_weight
        self.k_jasmin = k_jasmin

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    @staticmethod
    def _compute_topk_accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int,
    ):
        k = min(k, logits.shape[-1])
        topk = torch.topk(logits, k, dim=-1).indices

        return (
            topk
            .eq(labels.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
        )

    @staticmethod
    def _hard_labels(labels: torch.Tensor) -> torch.Tensor:
        """
        Supports both:
            hard labels: [B]
            soft / one-hot labels: [B, C]
        """
        if labels.ndim == 1:
            return labels

        return labels.argmax(dim=-1)

    @staticmethod
    def _zero_like_loss(reference: torch.Tensor):
        return reference.new_zeros(())

    def _compute_loss(self, outputs):
        """
        Faithful bViT:
            loss = cross_entropy

        Optional extension:
            loss = cross_entropy + jasmin_weight * jasmin_loss
        """
        jasmin_loss = (
            outputs.jasmin_loss
            if outputs.jasmin_loss is not None
            else self._zero_like_loss(outputs.cross_entropy)
        )

        loss = (
            outputs.cross_entropy
            + self.jasmin_weight * jasmin_loss
        )

        return loss, jasmin_loss

    def _compute_accuracies(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
    ):
        acc_1 = (
            torch.argmax(logits, dim=-1) == hard_labels
        ).float().mean()

        acc_3 = self._compute_topk_accuracy(
            logits,
            hard_labels,
            3,
        )

        acc_5 = self._compute_topk_accuracy(
            logits,
            hard_labels,
            5,
        )

        return acc_1, acc_3, acc_5

    def _log_recurrent_metrics(
        self,
        prefix: str,
        outputs,
        on_step: bool,
    ):
        """
        For bViT, `nfe` is the number of evaluations of the single shared
        Transformer block. For a standard bViT-B/S model this is normally 12.
        """
        if outputs.nfe is not None:
            self.log(
                f"{prefix}/nfe",
                float(outputs.nfe),
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        labels = batch["labels"]

        hard_labels = batch.get(
            "hard_labels",
            self._hard_labels(labels),
        )

        # The original bViT does not use JaSMin. Dense attention trajectories
        # are therefore only constructed when jasmin_weight > 0.
        use_jasmin = self.jasmin_weight > 0.0
        outputs = self.model(
            **inputs,
            labels=labels,
            regularize=use_jasmin,
            k_jasmin=self.k_jasmin,
            output_attentions=False,
            output_attentions_trajectory=False,
        )

        loss, jasmin_loss = self._compute_loss(outputs)

        acc_1, acc_3, acc_5 = self._compute_accuracies(
            outputs.logits,
            hard_labels,
        )

        self.log(
            "train/loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

        self.log(
            "train/loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "train/cross_entropy",
            outputs.cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if use_jasmin:
            self.log(
                "train/jasmin_loss",
                jasmin_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.log(
            "train/acc_1_step",
            acc_1,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

        self.log(
            "train/acc_1",
            acc_1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "train/acc_3",
            acc_3,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "train/acc_5",
            acc_5,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self._log_recurrent_metrics(
            "train",
            outputs,
            on_step=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        labels = batch["labels"]

        hard_labels = batch.get(
            "hard_labels",
            self._hard_labels(labels),
        )

        use_jasmin = self.jasmin_weight > 0.0

        outputs = self.model(
            **inputs,
            labels=labels,
            regularize=use_jasmin,
            k_jasmin=self.k_jasmin,
            output_attentions=False,
            output_attentions_trajectory=False,
        )

        loss, jasmin_loss = self._compute_loss(outputs)

        acc_1, acc_3, acc_5 = self._compute_accuracies(
            outputs.logits,
            hard_labels,
        )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/cross_entropy",
            outputs.cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if use_jasmin:
            self.log(
                "val/jasmin_loss",
                jasmin_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.log(
            "val/acc_1",
            acc_1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/acc_3",
            acc_3,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "val/acc_5",
            acc_5,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self._log_recurrent_metrics(
            "val",
            outputs,
            on_step=False,
        )

        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        labels = batch["labels"]

        hard_labels = batch.get(
            "hard_labels",
            self._hard_labels(labels),
        )

        outputs = self.model(
            **inputs,
            labels=labels,
            regularize=False,
            output_attentions=False,
            output_attentions_trajectory=False,
        )

        acc_1, acc_3, acc_5 = self._compute_accuracies(
            outputs.logits,
            hard_labels,
        )

        self.log(
            "test/cross_entropy",
            outputs.cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "test/acc_1",
            acc_1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/acc_3",
            acc_3,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "test/acc_5",
            acc_5,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self._log_recurrent_metrics(
            "test",
            outputs,
            on_step=False,
        )

        return outputs.cross_entropy

    def predict_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]

        outputs = self.model(
            **inputs,
            regularize=False,
            output_attentions=False,
            output_attentions_trajectory=False,
        )

        return torch.argmax(
            outputs.logits,
            dim=-1,
        )
