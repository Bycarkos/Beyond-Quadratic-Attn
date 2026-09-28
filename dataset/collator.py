import torch  # type: ignore
import numpy as np
from typing import Optional

import torch


def smooth_one_hot(labels, num_classes: int, smoothing: float = 0.0):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    targets = torch.full(
        (labels.shape[0], num_classes),
        fill_value=off_value,
        device=labels.device,
        dtype=torch.float32,
    )
    targets.scatter_(1, labels[:, None], on_value)
    return targets


def sample_beta(alpha: float, device):
    if alpha <= 0:
        return torch.tensor(1.0, device=device)
    return torch.distributions.Beta(alpha, alpha).sample().to(device)


def rand_bbox(width: int, height: int, lam: torch.Tensor, device):
    cut_ratio = torch.sqrt(1.0 - lam)
    cut_w = torch.clamp((width * cut_ratio).long(), min=1, max=width)
    cut_h = torch.clamp((height * cut_ratio).long(), min=1, max=height)

    cx = torch.randint(width, size=(1,), device=device).item()
    cy = torch.randint(height, size=(1,), device=device).item()

    x1 = max(cx - cut_w.item() // 2, 0)
    y1 = max(cy - cut_h.item() // 2, 0)
    x2 = min(cx + cut_w.item() // 2, width)
    y2 = min(cy + cut_h.item() // 2, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    images,
    labels,
    num_classes: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.5,
    mixup_prob: float = 1.0,
    label_smoothing: float = 0.0,
):
    if num_classes is None:
        raise ValueError("num_classes must be set when using MixUp or CutMix.")

    hard_labels = labels
    labels = smooth_one_hot(labels, num_classes, smoothing=label_smoothing)

    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, labels, hard_labels

    if torch.rand((), device=images.device).item() > mixup_prob:
        return images, labels, hard_labels

    batch_size = images.shape[0]
    perm = torch.randperm(batch_size, device=images.device)
    use_cutmix = cutmix_alpha > 0 and (
        mixup_alpha <= 0 or torch.rand((), device=images.device).item() < cutmix_prob
    )

    if use_cutmix:
        lam = sample_beta(cutmix_alpha, images.device)
        _, _, height, width = images.shape
        x1, y1, x2, y2 = rand_bbox(width, height, lam, images.device)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - area / float(width * height)
    else:
        lam = sample_beta(mixup_alpha, images.device)
        mixed_images = lam * images + (1.0 - lam) * images[perm]

    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_images, mixed_labels, hard_labels

class Collator:
    def __init__(
        self,
        processor,
        pad_token: Optional[int] = -100,
        transform=None,
        use_processor: bool = False,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 1.0,
        label_smoothing: float = 0.0,
        num_classes: Optional[int] = None,
        train: bool = True,
    ):
        self.processor = processor
        self._pad_token = pad_token
        self.transform = transform
        self.use_processor = use_processor
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.train = train

    def classification_collate_fn(self, batch):
        pixel_values = [
            item["image"].convert("RGB") if isinstance(item, dict)
            else item[0].convert("RGB")
            for item in batch
        ]

        labels = [
            item["label"] if isinstance(item, dict)
            else item[1]
            for item in batch
        ]

        images = [
            item["image"] if isinstance(item, dict)
            else item[0]
            for item in batch
        ]

        # --------------------------------------------------
        # 1. Per-image data augmentation
        # RandAugment / crop / flip happen here
        # --------------------------------------------------
        if self.transform:
            pixel_values = [self.transform(image) for image in pixel_values]

        # --------------------------------------------------
        # 2. Processor converts images to tensor batch
        # -------------------------------------------------- 
        if self.use_processor:
            processed_pixels = self.processor(
                pixel_values,
                return_tensors="pt",
            )
        else:
            processed_pixels = {
                "pixel_values": torch.stack(pixel_values, dim=0),
            }

        labels = torch.tensor(labels, dtype=torch.long)
        hard_labels = labels

        # --------------------------------------------------
        # 3. MixUp/CutMix happens after batching
        # --------------------------------------------------
        if self.train and (self.mixup_alpha > 0 or self.cutmix_alpha > 0):
            mixed_pixels, mixed_labels, hard_labels = apply_mixup_cutmix(
                processed_pixels["pixel_values"],
                labels,
                num_classes=self.num_classes,
                mixup_alpha=self.mixup_alpha,
                cutmix_alpha=self.cutmix_alpha,
                cutmix_prob=self.cutmix_prob,
                mixup_prob=self.mixup_prob,
                label_smoothing=self.label_smoothing,
            )

            processed_pixels["pixel_values"] = mixed_pixels
            labels = mixed_labels

        return {
            "pixel_values": processed_pixels,
            "labels": labels,
            "hard_labels": hard_labels,
            "raw_images": images,
        }

    def detector_collate_fn(self, batch):
        pixel_values = [item[0]["image"].unsqueeze(0) for item in batch]
        pixel_values = torch.cat(pixel_values, dim=0)

        encoding = self.processor.pad(pixel_values, return_tensors="pt")

        targets = [
            {
                "labels": torch.from_numpy(np.array(item[0]["labels"])),
                "boxes": torch.from_numpy(np.array(item[0]["bboxes"])),
            }
            for item in batch
        ]

        metadata = [item[-1] for item in batch]

        images = [item[1] for item in batch]

        return dict(
            pixel_values=encoding["pixel_values"],
            pixel_mask=encoding["pixel_mask"],
            targets=targets,
            metadata=metadata,
            images=images,
        )

    def ocr_collate_fn(self, batch):
        """
        Collate function for OCR dataset to handle image and text batching.

        Args:
            batch: List of dictionaries containing samples

        Returns:
            Dictionary of batched tensors and metadata
        """
        # Extract values from batch
        pixel_values = [item["pixel_values"].permute(2, 0, 1) for item in batch]
        text = [item["text"] for item in batch]

        tokens = [item.get("tokens", [])["input_ids"] for item in batch]
        images = [
            item.get("image", None) for item in batch
        ]  # Assuming raw_images should be from "image" key

        # Store original shapes before processing
        original_shapes = [value.shape for value in pixel_values]

        # Process images in one batch operation
        processed_pixels = self.processor(pixel_values, return_tensors="pt")

        # Get max sequence length for efficient padding
        max_seq_len = max(len(t) for t in tokens) if tokens else 0

        # Create padded token tensor efficiently
        if max_seq_len > 0:
            # Initialize with pad token
            padded_tokens = torch.full(
                size=(len(batch), max_seq_len),
                fill_value=self._pad_token,
                dtype=torch.long,  # Specify dtype explicitly
            )

            # Fill in actual token values
            for idx, seq in enumerate(tokens):
                if len(seq) > 0:  # Check if sequence is not empty
                    padded_tokens[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        else:
            # Handle case with empty tokens
            padded_tokens = torch.zeros((len(batch), 0), dtype=torch.long)

        # Store processed shapes for reference
        processed_shapes = [p_val.shape for p_val in processed_pixels["pixel_values"]]

        return {
            "pixel_values": processed_pixels,
            "text": text,
            "tokens": padded_tokens,
            "original_shapes": original_shapes,
            "resized_shapes": processed_shapes,
            "unpadded_tokens": [torch.tensor(t, dtype=torch.long) for t in tokens],
            "raw_images": images,
        }
