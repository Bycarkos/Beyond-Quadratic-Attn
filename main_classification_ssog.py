"""Train SSOG-ViT using the supervised classification pipeline."""

import argparse

from hydra import compose, initialize

from main_classification_scratch import main as run_classification
from models.SSOGViT import SSOG_ViT
from trainers.loss_trainer_ssog_classification import (
    TrainerSSOGClassification,
)


def main(cfg):
    run_classification(
        cfg=cfg,
        model_class=SSOG_ViT,
        trainer_class=TrainerSSOGClassification,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="SSOG classification YAML file from configs/classification.",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="configs/classification"):
        config = compose(config_name=args.config_file)
    main(config)

