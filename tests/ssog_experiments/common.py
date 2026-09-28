"""Shared helpers for SSOG tests and visualization experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGENET_CACHE = Path(
    os.environ.get("IMAGENET_CACHE", "~/imagenet-1k")
).expanduser()

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
IMAGENET_DENORMALIZE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_DENORMALIZE_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def find_imagenet_arrow_shards(
    cache_dir: Path = DEFAULT_IMAGENET_CACHE,
    split: str = "validation",
) -> list[Path]:
    """Find prepared Hugging Face ImageNet shards without contacting the Hub."""
    pattern = f"imagenet-1k-{split}-*.arrow"
    return sorted(cache_dir.rglob(pattern))


def _as_rgb_pil(image: Any):
    if hasattr(image, "convert"):
        return image.convert("RGB")

    # Depending on the datasets/Pillow versions, a raw Arrow image can be
    # represented as {"bytes": ..., "path": ...}.
    if isinstance(image, dict):
        from io import BytesIO

        from PIL import Image

        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if image.get("path") is not None:
            return Image.open(image["path"]).convert("RGB")

    raise TypeError(f"Unsupported ImageNet image value: {type(image)!r}")


def imagenet_eval_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Lambda(_as_rgb_pil),
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )


def load_cached_imagenet_batch(
    batch_size: int = 2,
    img_size: int = 224,
    cache_dir: Path = DEFAULT_IMAGENET_CACHE,
    split: str = "validation",
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Load one batch from the first local Arrow shard, with no network access."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for this test.") from exc

    shards = find_imagenet_arrow_shards(cache_dir, split)
    if not shards:
        raise FileNotFoundError(
            f"No {split!r} ImageNet Arrow shards found below {cache_dir}. "
            "Set IMAGENET_CACHE to the Hugging Face cache root."
        )

    dataset = Dataset.from_file(str(shards[0]))
    transform = imagenet_eval_transform(img_size)
    images = []
    labels = []
    raw_images = []

    for index in range(batch_size):
        sample = dataset[index]
        raw_image = _as_rgb_pil(sample["image"])
        raw_images.append(raw_image.copy())
        images.append(transform(raw_image))
        labels.append(int(sample["label"]))

    return (
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        raw_images,
    )


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_DENORMALIZE_MEAN.to(image)
    std = IMAGENET_DENORMALIZE_STD.to(image)
    return (image * std + mean).clamp(0, 1)

