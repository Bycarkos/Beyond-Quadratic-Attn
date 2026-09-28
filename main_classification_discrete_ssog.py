"""Train the discrete SSOG baseline with the shared supervised pipeline."""

import argparse

from hydra import compose, initialize

from main_classification_scratch import main as run_classification
from models.DiscreteSSOGViT import DiscreteSSOG_ViT
from trainers.loss_trainer_discrete_ssog_classification import (
    TrainerDiscreteSSOGClassification,
)


def main(cfg):
    run_classification(
        cfg=cfg,
        model_class=DiscreteSSOG_ViT,
        trainer_class=TrainerDiscreteSSOGClassification,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", required=True,
        help="Discrete SSOG YAML file from configs/classification.",
    )
    args = parser.parse_args()
    with initialize(version_base=None, config_path="configs/classification"):
        config = compose(config_name=args.config_file)
    main(config)
