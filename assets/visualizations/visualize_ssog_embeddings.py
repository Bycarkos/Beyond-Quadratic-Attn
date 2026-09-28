"""PCA/t-SNE analysis of SSOG representations on CIFAR-100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from models.SSOGViT import SSOG_ViT


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def compose_config(name):
    with initialize(version_base=None, config_path="../../configs/classification"):  # relative to this file
        return compose(config_name=name)


def load_model(cfg, checkpoint_path, device):
    model = SSOG_ViT(**cfg.modeling.inputs)
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False
    )
    state = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.removeprefix("model.") in model_state
        and model_state[key.removeprefix("model.")].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


def balanced_indices(targets, samples_per_class, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    selected = []
    for class_index in np.unique(targets):
        candidates = np.flatnonzero(targets == class_index)
        selected.extend(
            rng.choice(
                candidates,
                size=min(samples_per_class, len(candidates)),
                replace=False,
            ).tolist()
        )
    rng.shuffle(selected)
    return selected


@torch.inference_mode()
def extract_features(model, loader, device):
    features, labels = [], []
    for images, batch_labels in loader:
        output = model(
            pixel_values=images.to(device, non_blocking=True),
            output_attentions=False,
            output_attentions_trajectory=False,
            regularize=False,
        )
        final_state = output.hidden_states[-1]
        representation = (
            final_state[:, 0] if model.has_cls_token else final_state.mean(dim=1)
        )
        features.append(representation.float().cpu())
        labels.append(batch_labels)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def cluster_metrics(features, labels, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    split = int(0.8 * len(order))
    train_indices, test_indices = order[:split], order[split:]
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(features[train_indices], labels[train_indices])
    return {
        "silhouette_cosine": float(
            silhouette_score(features, labels, metric="cosine")
        ),
        "knn_5_holdout_accuracy": float(
            knn.score(features[test_indices], labels[test_indices])
        ),
        "num_samples": int(len(features)),
        "num_classes": int(len(np.unique(labels))),
        "feature_dimension": int(features.shape[1]),
    }


def scatter_plot(points, labels, class_names, title, output_path):
    figure, axis = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("turbo", len(class_names))
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap=cmap,
        s=10,
        alpha=0.72,
        linewidths=0,
    )
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.15)
    colorbar = figure.colorbar(scatter, ax=axis, ticks=np.arange(0, 100, 10))
    colorbar.set_label("CIFAR-100 fine-class index")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def selected_classes_plot(points, labels, class_names, class_indices, title, path):
    mask = np.isin(labels, class_indices)
    figure, axis = plt.subplots(figsize=(13, 10))
    colors = plt.get_cmap("tab20", len(class_indices))
    for color_index, class_index in enumerate(class_indices):
        class_mask = mask & (labels == class_index)
        axis.scatter(
            points[class_mask, 0],
            points[class_mask, 1],
            s=18,
            alpha=0.76,
            color=colors(color_index),
            label=class_names[class_index],
            linewidths=0,
        )
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.15)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = compose_config(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (cfg.modeling.inputs.img_size, cfg.modeling.inputs.img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = CIFAR100(
        root=str(Path(cfg.data.dataset.dataset_path).expanduser()),
        train=False,
        download=False,
        transform=transform,
    )
    indices = balanced_indices(dataset.targets, args.samples_per_class, args.seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    features, labels = extract_features(model, loader, device)
    normalized = StandardScaler().fit_transform(features)
    pca_model = PCA(n_components=min(50, normalized.shape[1]), random_state=args.seed)
    pca_features = pca_model.fit_transform(normalized)
    pca_2d = pca_features[:, :2]
    tsne_2d = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=args.tsne_iterations,
        random_state=args.seed,
    ).fit_transform(pca_features)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_classes = np.linspace(0, 99, args.legend_classes, dtype=int)
    scatter_plot(pca_2d, labels, dataset.classes, "SSOG linear-probe features — PCA", output_dir / "pca_all_classes.png")
    scatter_plot(tsne_2d, labels, dataset.classes, "SSOG linear-probe features — t-SNE", output_dir / "tsne_all_classes.png")
    selected_classes_plot(pca_2d, labels, dataset.classes, selected_classes, "PCA — representative classes", output_dir / "pca_selected_classes.png")
    selected_classes_plot(tsne_2d, labels, dataset.classes, selected_classes, "t-SNE — representative classes", output_dir / "tsne_selected_classes.png")

    metrics = cluster_metrics(features, labels, args.seed)
    metrics.update(
        {
            "pca_2d_explained_variance": float(
                pca_model.explained_variance_ratio_[:2].sum()
            ),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "model_config": args.model_config,
        }
    )
    with (output_dir / "cluster_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        features=features,
        labels=labels,
        pca=pca_2d,
        tsne=tsne_2d,
        dataset_indices=np.asarray(indices),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument("--legend-classes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
