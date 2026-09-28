import argparse
import math
from operator import call
from pathlib import Path
import warnings

import torch
import wandb
from dataset.collator import Collator
from hydra import compose, initialize
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EMAWeightAveraging,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    Timer,
)
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
)
from lightning.pytorch.loggers import WandbLogger

from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise

from models.ODEViT import ODE_ViT
from muon import SingleDeviceMuonWithAuxAdam
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import ImageFile
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.imagenet import ImageFolder
from trainers.loss_trainer_classification import TrainerClassification
from transformers import ViTImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)


torch.set_float32_matmul_precision("medium")


class ResumeCurrentModelState(Callback):
    """Restore live weights instead of the EMA state stored as state_dict."""

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        del trainer
        current_model_state = checkpoint.get("current_model_state")
        if current_model_state is None:
            raise KeyError(
                "resume_use_current_model_state=true, but the checkpoint has "
                "no current_model_state."
            )
        pl_module.load_state_dict(current_model_state)
        print("Restored live current_model_state from the checkpoint.")


class ResumeLearningRateOverride(Callback):
    """Replace the restored scheduler base LR after a full-state resume."""

    def __init__(self, base_lr: float):
        super().__init__()
        if base_lr <= 0:
            raise ValueError("resume_base_lr must be positive.")
        self.base_lr = float(base_lr)

    def on_train_start(self, trainer, pl_module):
        del pl_module
        for scheduler_config in trainer.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            if not isinstance(scheduler, LambdaLR):
                raise TypeError(
                    "resume_base_lr currently supports the LambdaLR training "
                    f"scheduler, received {type(scheduler).__name__}."
                )

            scheduler.base_lrs = [self.base_lr] * len(scheduler.base_lrs)
            resumed_lrs = [
                self.base_lr * lr_lambda(scheduler.last_epoch)
                for lr_lambda in scheduler.lr_lambdas
            ]
            for param_group, resumed_lr in zip(
                scheduler.optimizer.param_groups, resumed_lrs
            ):
                param_group["initial_lr"] = self.base_lr
                param_group["lr"] = resumed_lr
            scheduler._last_lr = resumed_lrs
            print(
                "Overrode resumed scheduler base LR with "
                f"{self.base_lr:g}; current LR is {resumed_lrs}."
            )


def _cfg_value(cfg, key, default=None):
    if key not in cfg:
        return default
    if OmegaConf.is_missing(cfg, key):
        return default
    return cfg[key]


def resolve_resume_checkpoint(cfg: DictConfig):
    """Return the checkpoint to resume, or None for a fresh training run."""
    experiment_cfg = cfg.setup.experiment
    if not bool(_cfg_value(experiment_cfg, "recover_from_checkpoint", False)):
        return None

    configured_path = _cfg_value(experiment_cfg, "resume_checkpoint_path")
    if configured_path is None:
        configured_path = (
            Path(str(cfg.modeling.checkpoint_path)).expanduser() / "last.ckpt"
        )
    else:
        configured_path = Path(str(configured_path)).expanduser()

    configured_path = configured_path.resolve()
    if not configured_path.is_file():
        raise FileNotFoundError(
            "Checkpoint recovery was requested, but the checkpoint does not "
            f"exist: {configured_path}"
        )

    print(f"Resuming training from checkpoint: {configured_path}")
    return str(configured_path)


def initialize_from_pretrained_checkpoint(model, cfg: DictConfig):
    """Load compatible model weights without restoring training state.

    This is intentionally separate from ``resolve_resume_checkpoint``: transfer
    learning starts a new optimizer/scheduler and skips tensors whose shapes do
    not match (normally the ImageNet classifier).
    """
    experiment_cfg = cfg.setup.experiment
    configured_path = _cfg_value(experiment_cfg, "pretrained_checkpoint_path")
    if configured_path is None:
        return

    checkpoint_path = Path(str(configured_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint does not exist: {checkpoint_path}"
        )

    # These are trusted checkpoints produced locally by this training pipeline.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for checkpoint_key, tensor in state_dict.items():
        model_key = checkpoint_key.removeprefix("model.")
        if model_key in model_state and model_state[model_key].shape == tensor.shape:
            compatible[model_key] = tensor
        elif model_key in model_state:
            skipped.append(
                f"{model_key}: checkpoint {tuple(tensor.shape)} != model "
                f"{tuple(model_state[model_key].shape)}"
            )

    if not compatible:
        raise RuntimeError(
            f"No compatible model tensors were found in {checkpoint_path}"
        )

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Initialized {len(compatible)} tensors from: {checkpoint_path}")
    if skipped:
        print("Skipped shape-mismatched tensors:\n  " + "\n  ".join(skipped))
    if unexpected:
        raise RuntimeError(f"Unexpected pretrained keys: {unexpected}")
    non_classifier_missing = [
        key for key in missing if not key.startswith("classifier.")
    ]
    if non_classifier_missing:
        raise RuntimeError(
            "Pretrained checkpoint is missing non-classifier tensors: "
            f"{non_classifier_missing}"
        )

    if bool(_cfg_value(experiment_cfg, "freeze_pretrained_backbone", False)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")
        print("Frozen pretrained backbone; only classifier parameters are trainable.")


def resolve_num_devices(devices) -> int:
    if devices is None:
        return 1
    if isinstance(devices, int):
        if devices == -1:
            return max(1, torch.cuda.device_count())
        return max(1, devices)
    if isinstance(devices, str):
        if devices in {"auto", "-1"}:
            return max(1, torch.cuda.device_count())
        if "," in devices:
            return max(1, len([d for d in devices.split(",") if d.strip()]))
        if devices.isdigit():
            return max(1, int(devices))
        return 1
    if hasattr(devices, "__len__"):
        return max(1, len(devices))
    return 1


def compute_optimizer_steps_per_epoch(
    num_samples: int,
    per_device_batch_size: int,
    num_devices: int,
    accumulation_steps: int,
    drop_last: bool,
) -> int:
    global_batch_size = per_device_batch_size * num_devices
    if drop_last:
        micro_batches = num_samples // global_batch_size
    else:
        micro_batches = math.ceil(num_samples / global_batch_size)
    micro_batches = max(1, micro_batches)
    return max(1, math.ceil(micro_batches / accumulation_steps))


def resolve_scheduler_steps(cfg: DictConfig, train_dataset) -> tuple[int, int, int]:
    scheduler_cfg = cfg.setup.scheduler
    train_loader_cfg = cfg.data.collator.train
    accumulation_steps = (
        cfg.setup.experiment.accumulation_steps
        if cfg.setup.experiment.accumulation_steps > 0
        else 1
    )
    num_devices = resolve_num_devices(cfg.setup.hardware.devices)
    steps_per_epoch = compute_optimizer_steps_per_epoch(
        num_samples=len(train_dataset),
        per_device_batch_size=train_loader_cfg.batch_size,
        num_devices=num_devices,
        accumulation_steps=accumulation_steps,
        drop_last=bool(train_loader_cfg.drop_last),
    )

    warmup_steps = _cfg_value(scheduler_cfg, "warmup_steps")
    if warmup_steps is None:
        warmup_epochs = _cfg_value(scheduler_cfg, "warmup_epochs")
        if warmup_epochs is None:
            legacy_warmup = _cfg_value(scheduler_cfg, "num_warmup_steps", 0)
            if legacy_warmup <= cfg.setup.experiment.epochs:
                warmup_epochs = legacy_warmup
            else:
                warmup_steps = legacy_warmup
        if warmup_steps is None:
            warmup_steps = int(round(float(warmup_epochs) * steps_per_epoch))

    decay_steps = _cfg_value(scheduler_cfg, "decay_steps")
    if decay_steps is None:
        decay_epochs = _cfg_value(scheduler_cfg, "decay_epochs")
        if decay_epochs is not None:
            decay_steps = int(round(float(decay_epochs) * steps_per_epoch))
        else:
            decay_steps = _cfg_value(scheduler_cfg, "num_training_steps")

    if decay_steps is None:
        decay_steps = cfg.setup.experiment.epochs * steps_per_epoch

    return steps_per_epoch, int(warmup_steps), int(decay_steps)


def get_warmup_scheduler(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    scheduler_type="polynomial",
    power=0.9,
    lr_end_factor=0.0,
    min_lr=None,
    warmup_start_factor=0.0,
):
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "linear":
        scheduler_type = "polynomial"
        power = 1.0
    if scheduler_type not in {"polynomial", "cosine", "constant"}:
        raise ValueError(
            "scheduler.type must be one of: polynomial, linear, cosine, constant"
        )

    if min_lr is not None:
        min_lr_factor = max(
            float(min_lr) / float(group["lr"])
            for group in optimizer.param_groups
            if group["lr"] > 0
        )
    else:
        min_lr_factor = float(lr_end_factor)
    min_lr_factor = min(max(min_lr_factor, 0.0), 1.0)
    warmup_start_factor = min(max(float(warmup_start_factor), 0.0), 1.0)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            progress = float(current_step) / float(max(1, num_warmup_steps))
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress

        if scheduler_type == "constant":
            return 1.0

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        progress = min(progress, 1.0)

        if scheduler_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay = (1.0 - progress) ** power

        return min_lr_factor + (1.0 - min_lr_factor) * decay

    return LambdaLR(optimizer, lr_lambda)


def build_muon_optimizer(
    flow_model,
    lr=5e-4,
    weight_decay=5e-2,
    muon_lr=None,
    betas=(0.9, 0.95),
    eps=1e-8,
):
    """
    Use Muon only for hidden transformer matrix weights.

    Embeddings, adapters, classifier weights, normalization parameters, and
    biases use the optimizer's auxiliary AdamW update. Every trainable
    parameter must belong to exactly one group.
    """
    muon_params = []
    adam_params = []

    for name, p in flow_model.named_parameters():
        if not p.requires_grad:
            continue

        is_transformer_hidden_weight = p.ndim == 2

        if is_transformer_hidden_weight:
            muon_params.append(p)
        else:
            adam_params.append(p)

    param_groups = []
    if muon_params:
        param_groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr if muon_lr is not None else lr,
                weight_decay=weight_decay,
            )
        )
    if adam_params:
        param_groups.append(
            dict(
                params=adam_params,
                use_muon=False,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        )

    grouped_param_ids = [id(p) for group in param_groups for p in group["params"]]
    trainable_param_ids = [id(p) for p in flow_model.parameters() if p.requires_grad]
    if len(grouped_param_ids) != len(set(grouped_param_ids)):
        raise ValueError(
            "A trainable parameter appears in more than one optimizer group."
        )
    if set(grouped_param_ids) != set(trainable_param_ids):
        raise ValueError("Some trainable parameters are missing from the optimizer.")

    return SingleDeviceMuonWithAuxAdam(param_groups)

def build_optimizer(flow_model, optimizer_cfg):
    """Build the configured optimizer while preserving Muon as the default."""
    optimizer_type = str(_cfg_value(optimizer_cfg, "type", "muon")).lower()
    if optimizer_type == "muon":
        kwargs = {
            key: value
            for key, value in dict(optimizer_cfg).items()
            if key != "type"
        }
        return build_muon_optimizer(flow_model, **kwargs)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            flow_model.parameters(),
            lr=float(_cfg_value(optimizer_cfg, "lr", 5e-4)),
            weight_decay=float(_cfg_value(optimizer_cfg, "weight_decay", 0.05)),
            betas=tuple(_cfg_value(optimizer_cfg, "betas", (0.9, 0.999))),
            eps=float(_cfg_value(optimizer_cfg, "eps", 1e-8)),
        )
    raise ValueError(
        f"Unsupported optimizer type {optimizer_type!r}; "
        "expected 'muon' or 'adamw'."
    )


def main(
    cfg: DictConfig,
    model_class=ODE_ViT,
    trainer_class=TrainerClassification,
):
    seed = int(_cfg_value(cfg.setup.experiment, "seed", 42))
    seed_everything(seed, workers=True)

    dataset_download = cfg.data.dataset.get("download", False)

    if cfg.data.dataset.name == "cifar100":
        train_dataset = CIFAR100(
            root=cfg.data.dataset.dataset_path, download=dataset_download, train=True
        )
        validation_dataset = CIFAR100(
            root=cfg.data.dataset.dataset_path, download=dataset_download, train=False
        )
    elif cfg.data.dataset.name == "cifar10":
        train_dataset = CIFAR10(
            root=cfg.data.dataset.dataset_path, download=dataset_download, train=True
        )
        validation_dataset = CIFAR10(
            root=cfg.data.dataset.dataset_path, download=dataset_download, train=False
        )
    elif cfg.data.dataset.name == "inaturalist":
        train_dataset = ImageFolder(root=cfg.data.dataset.dataset_path + "/train")
        validation_dataset = ImageFolder(root=cfg.data.dataset.dataset_path + "/val")

    else:
        from datasets import load_dataset

        ds = load_dataset(
            cfg.data.dataset.name, cache_dir=cfg.data.dataset.dataset_path
        )
        train_dataset = ds["train"]
        validation_dataset = ds["validation"]

    processor = None #ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    augmentation_cfg = cfg.data.get("augmentation", {})
    crop_scale = tuple(augmentation_cfg.get("crop_scale", [0.4, 1.0]))
    erasing_scale = tuple(augmentation_cfg.get("random_erasing_scale", [0.02, 0.2]))
    erasing_ratio = tuple(augmentation_cfg.get("random_erasing_ratio", [0.3, 3.3]))
    train_transforms = [
        transforms.RandomResizedCrop(
            (cfg.modeling.inputs.img_size, cfg.modeling.inputs.img_size),
            scale=crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(
            p=float(augmentation_cfg.get("horizontal_flip_prob", 0.5))
        ),
    ]
    if bool(augmentation_cfg.get("randaugment_enabled", True)):
        train_transforms.append(
            transforms.RandAugment(
                num_ops=int(augmentation_cfg.get("randaugment_num_ops", 2)),
                magnitude=int(augmentation_cfg.get("randaugment_magnitude", 9)),
            )
        )
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    random_erasing_prob = float(
        augmentation_cfg.get("random_erasing_prob", 0.25)
    )
    if random_erasing_prob > 0:
        train_transforms.append(
            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=erasing_scale,
                ratio=erasing_ratio,
                value=augmentation_cfg.get("random_erasing_value", "random"),
            )
        )
    transform_train = transforms.Compose(train_transforms)
    
        # Validation transform (no augmentation, only resizing to 224x224)
    transform_val = transforms.Compose([
            transforms.Resize((cfg.modeling.inputs.img_size, cfg.modeling.inputs.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    collator_train = Collator(
        processor=processor,
        use_processor=False,
        transform=transform_train,
        train=True,
        num_classes=cfg.data.dataset.num_classes,
        mixup_alpha=augmentation_cfg.get("mixup_alpha", 0.0),
        cutmix_alpha=augmentation_cfg.get("cutmix_alpha", 0.0),
        cutmix_prob=augmentation_cfg.get("cutmix_prob", 0.5),
        mixup_prob=augmentation_cfg.get("mixup_prob", 1.0),
        label_smoothing=augmentation_cfg.get("label_smoothing", 0.0),
    )
    collator_val = Collator(
        processor=processor,
        use_processor=False,
        transform=transform_val,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        **cfg.data.collator.train,
        collate_fn=collator_train.classification_collate_fn,
    )
    val_dataloader = DataLoader(
        validation_dataset,
        **cfg.data.collator.val,
        collate_fn=collator_val.classification_collate_fn,
    )

    
    model = model_class(**cfg.modeling.inputs)
    initialize_from_pretrained_checkpoint(model, cfg)
    if cfg.modeling.get("compile", False):
        model = torch.compile(model)

    optimizer = build_optimizer(model, cfg.setup.optimizer)

    accumulate_grad_batches = (
        cfg.setup.experiment.accumulation_steps
        if cfg.setup.experiment.accumulation_steps > 0
        else 1
    )
    steps_per_epoch, num_warmup_steps, num_training_steps = resolve_scheduler_steps(
        cfg, train_dataset
    )
    scheduler_type = _cfg_value(cfg.setup.scheduler, "type", "polynomial")

    with open_dict(cfg):
        cfg.setup.scheduler.resolved_steps_per_epoch = steps_per_epoch
        cfg.setup.scheduler.resolved_num_warmup_steps = num_warmup_steps
        cfg.setup.scheduler.resolved_num_training_steps = num_training_steps

    scheduler = get_warmup_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_type=scheduler_type,
        power=_cfg_value(cfg.setup.scheduler, "power", 0.9),
        lr_end_factor=_cfg_value(cfg.setup.scheduler, "lr_end_factor", 0.0),
        min_lr=_cfg_value(cfg.setup.scheduler, "min_lr"),
        warmup_start_factor=_cfg_value(cfg.setup.scheduler, "warmup_start_factor", 0.0),
    )

    trainer_kwargs = dict(cfg.setup.get("trainer", {}))
    lighning_model = trainer_class(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        **trainer_kwargs,
    )

    if cfg.log_wandb:
        wandb_logger = WandbLogger(
            **cfg.setup.wandb,
            settings=wandb.Settings(code_dir="./"),
            config=dict(cfg),
        )
        #wandb_logger.watch(lighning_model, log="gradients", log_freq=500)

    else:
        wandb_logger = False

    ## CALLBACKS

    timer = Timer(duration=dict(weeks=4, days=2))
    model_summary = ModelSummary(max_depth=2)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks = [timer, model_summary]

    if bool(
        _cfg_value(
            cfg.setup.experiment, "resume_use_current_model_state", False
        )
    ):
        if not bool(
            _cfg_value(
                cfg.setup.experiment, "recover_from_checkpoint", False
            )
        ):
            raise ValueError(
                "resume_use_current_model_state requires "
                "recover_from_checkpoint=true."
            )
        callbacks.append(ResumeCurrentModelState())

    resume_base_lr = _cfg_value(cfg.setup.experiment, "resume_base_lr")
    if resume_base_lr is not None:
        if not bool(
            _cfg_value(
                cfg.setup.experiment, "recover_from_checkpoint", False
            )
        ):
            raise ValueError(
                "resume_base_lr requires recover_from_checkpoint=true."
            )
        callbacks.append(ResumeLearningRateOverride(resume_base_lr))

    ema_cfg = cfg.setup.get("ema", {})
    if bool(ema_cfg.get("enabled", False)):
        callbacks.append(
            EMAWeightAveraging(decay=float(ema_cfg.get("decay", 0.999)))
        )

    if wandb_logger:
        callbacks.append(lr_monitor)


    checkpointer = ModelCheckpoint(
        monitor="val/acc_1",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename=f"{{epoch:03d}}-{cfg.modeling.checkpoint_name}.pt",
        dirpath=cfg.modeling.checkpoint_path,
        auto_insert_metric_name=False,
    )

    callbacks.append(checkpointer)

    checkpoint_cfg = cfg.setup.get("checkpoint", {})
    if bool(checkpoint_cfg.get("save_rolling", False)):
        callbacks.append(
            ModelCheckpoint(
                monitor=None,
                save_top_k=1,
                save_last=False,
                every_n_epochs=1,
                filename="rolling-last",
                dirpath=cfg.modeling.checkpoint_path,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    callbacks.append(progress_bar)

    ## Model

    trainer = Trainer(
        accelerator="auto",
        devices=cfg.setup.hardware.devices,
        strategy=_cfg_value(cfg.setup.hardware, "strategy", "auto"),
        precision=_cfg_value(cfg.setup.hardware, "precision", "32-true"),
        max_epochs=cfg.setup.experiment.epochs,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.setup.experiment.log_every,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=float(
            _cfg_value(cfg.setup.experiment, "gradient_clip_val", 1.0)
        ),
        gradient_clip_algorithm="norm",
    )

    resume_checkpoint = resolve_resume_checkpoint(cfg)
    trainer.fit(
        lighning_model,
        train_dataloader,
        val_dataloader,
        ckpt_path=resume_checkpoint,
        # Full-state loading is required for trusted local Lightning checkpoints.
        weights_only=False if resume_checkpoint is not None else None,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Yaml Config file with the configuration for the models to run, for now the models to run are [defoDeTR, DeTR, CondDetr, Yolo]",
    )

    args = parser.parse_args()

    with initialize(version_base=None, config_path="configs/classification"):
        cfg = compose(config_name=args.config_file)
        main(cfg)
