"""Train the dense-attention ODE-ViT classification baseline, with the same
qualitative attention-map tracking the SSOG trainer provides for its Gaussian
mixture (see trainers/loss_trainer_dense_classification.py)."""

import argparse

from hydra import compose, initialize

from main_classification_scratch import main as run_classification
from models.ODEViT import ODE_ViT
from trainers.loss_trainer_dense_classification import (
    TrainerDenseAttentionClassification,
)


def main(cfg):
    run_classification(
        cfg=cfg,
        model_class=ODE_ViT,
        trainer_class=TrainerDenseAttentionClassification,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Dense ODE-ViT YAML file from configs/classification.",
    )
    args = parser.parse_args()
    with initialize(version_base=None, config_path="configs/classification"):
        config = compose(config_name=args.config_file)
    main(config)
