from typing import Optional

import lightning as L
import torch


class TrainerClassification(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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
    def _compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int):
        k = min(k, logits.shape[-1])
        topk = torch.topk(logits, k, dim=-1).indices
        return topk.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean()

    @staticmethod
    def _hard_labels(labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 1:
            return labels
        return labels.argmax(dim=-1)

    @staticmethod
    def _zero_like_loss(reference: torch.Tensor):
        return reference.new_zeros(())

    def _log_optional_ode_metrics(self, prefix: str, outputs, on_step: bool):
        if outputs.nfe is not None:
            self.log(
                f"{prefix}/nfe",
                float(outputs.nfe),
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        if outputs.error_upper_bound is not None:
            self.log(
                f"{prefix}/error_upper_bound",
                outputs.error_upper_bound,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        labels = batch["labels"]
        hard_labels = batch.get("hard_labels", self._hard_labels(labels))

        outputs = self.model(
            **inputs, labels=labels, max_norm_bounding=10, regularize=True, k_jasmin=4
        )

        lyapunov_loss = (
            outputs.lyapunov_loss
            if outputs.lyapunov_loss is not None
            else self._zero_like_loss(outputs.cross_entropy)
        )
        jasmin_loss = (
            outputs.jasmin_loss
            if outputs.jasmin_loss is not None
            else self._zero_like_loss(outputs.cross_entropy)
        )

        loss = 1 * outputs.cross_entropy + 1e-2 * jasmin_loss
        #1e-2 * lyapunov_loss + 1e-2 * jasmin_loss

        acc_1 = (torch.argmax(outputs.logits, dim=-1) == hard_labels).float().mean()
        acc_3 = self._compute_topk_accuracy(outputs.logits, hard_labels, 3)
        acc_5 = self._compute_topk_accuracy(outputs.logits, hard_labels, 5)

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train/cross_entropy",
            outputs.cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            "train/lyapunov_loss",
            lyapunov_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            "train/jasmin_loss",
            jasmin_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )

        self.log("train/acc_1_step", acc_1, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        self.log("train/acc_1", acc_1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/acc_3", acc_3, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/acc_5", acc_5, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self._log_optional_ode_metrics("train", outputs, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        labels = batch["labels"]
        hard_labels = batch.get("hard_labels", self._hard_labels(labels))

        outputs = self.model(
            **inputs, labels=labels, max_norm_bounding=10, regularize=True, k_jasmin=4
        )

        jasmin_loss = (
            outputs.jasmin_loss
            if outputs.jasmin_loss is not None
            else self._zero_like_loss(outputs.cross_entropy)
        )
        loss = outputs.cross_entropy + 1e-3 * jasmin_loss

        acc_1 = (torch.argmax(outputs.logits, dim=-1) == hard_labels).float().mean()
        acc_3 = self._compute_topk_accuracy(outputs.logits, hard_labels, 3)
        acc_5 = self._compute_topk_accuracy(outputs.logits, hard_labels, 5)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True  )
        self.log(
            "val/cross_entropy",
            outputs.cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.log(
            "val/jasmin_loss",
            jasmin_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )

        self.log("val/acc_1", acc_1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc_3", acc_3, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/acc_5", acc_5, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self._log_optional_ode_metrics("val", outputs, on_step=False)

        return loss

    def predict_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        outputs = self.model(
            **inputs, max_norm_bounding=10, regularize=True, k_jasmin=4
        )
        return torch.argmax(outputs.logits, dim=-1)
