"""Use the existing SSOG diagnostics on the last discrete block."""

from trainers.loss_trainer_ssog_classification import TrainerSSOGClassification


class TrainerDiscreteSSOGClassification(TrainerSSOGClassification):
    """Classification metrics plus final-block Gaussian statistics and maps.

    SSOG diagnostic keys refer only to the last block, not averages over depth.
    The inherited attention-input hook captures that block's normalized input.
    """

    @property
    def ssog_attention(self):
        return self.model.blocks[-1].attn

    def _log_optional_ode_metrics(self, prefix, outputs, on_step):
        self.log(
            f"{prefix}/depth", float(self.model.depth),
            on_step=on_step, on_epoch=True, sync_dist=True,
        )
