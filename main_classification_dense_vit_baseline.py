"""Train the dense standard-attention ViT (MatchedViT) baseline using the
same "classification" trainer family as the SSOG and dense-ODEViT baselines
(TrainerClassification + attention-map visualization), instead of
TrainerBViTClassification.

Kept separate from main_classification_vit_baseline.py, which the ImageNet
matched-baseline jobs depend on with TrainerBViTClassification's specific
behavior (JaSMin off by default, no attention-map logging) -- swapping the
trainer there would change a script those in-progress runs rely on.
"""

import argparse

from hydra import compose, initialize

from main_classification_scratch import main as run_classification
from models.MatchedViT import MatchedViT
from trainers.loss_trainer_dense_classification import (
    TrainerDenseViTAttentionClassification,
)


def main(cfg):
    run_classification(
        cfg=cfg,
        model_class=MatchedViT,
        trainer_class=TrainerDenseViTAttentionClassification,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Dense ViT YAML file from configs/classification.",
    )
    args = parser.parse_args()
    with initialize(version_base=None, config_path="configs/classification"):
        config = compose(config_name=args.config_file)
    main(config)
