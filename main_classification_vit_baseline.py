"""Train the matched standard ViT ImageNet baseline."""

import argparse

from hydra import compose, initialize

from main_classification_scratch import main as run_classification
from models.MatchedViT import MatchedViT
from trainers.loss_trainer_classification_looped import (
    TrainerBViTClassification,
)


def main(cfg):
    run_classification(
        cfg=cfg,
        model_class=MatchedViT,
        trainer_class=TrainerBViTClassification,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Matched ViT YAML file from configs/classification.",
    )
    args = parser.parse_args()
    with initialize(version_base=None, config_path="configs/classification"):
        config = compose(config_name=args.config_file)
    main(config)

