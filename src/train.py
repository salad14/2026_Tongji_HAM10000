"""Train and evaluate HAM10000 models.

Examples:
    python src/train.py --experiment meta_only --epochs 20
    python src/train.py --experiment image_only --epochs 30 --batch-size 32 --amp
    python src/train.py --experiment fusion --epochs 30 --batch-size 24 --amp
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    HAM10000Dataset,
    MetadataEncoder,
    build_weighted_sampler,
    compute_class_weights,
    get_train_transform,
    get_valid_transform,
)
from src.models import build_model  # noqa: E402


LABEL_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@dataclass
class TrainConfig:
    experiment: str
    epochs: int
    batch_size: int
    image_size: int
    lr: float
    weight_decay: float
    num_workers: int
    pretrained: bool
    freeze_backbone: bool
    use_class_weights: bool
    use_weighted_sampler: bool
    amp: bool
    seed: int


class MetadataOnlyDataset(Dataset):
    """Dataset for metadata-only training, without reading image files."""

    def __init__(self, csv_path: str | Path, metadata_encoder: MetadataEncoder) -> None:
        self.df = pd.read_csv(csv_path)
        self.metadata_encoder = metadata_encoder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        metadata = self.metadata_encoder.encode(row)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return metadata, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HAM10000 classification models.")
    parser.add_argument(
        "--experiment",
        choices=["meta_only", "image_only", "fusion"],
        required=True,
        help="Which ablation experiment to run.",
    )
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load outputs/<experiment>/best_model.pth and recompute test metrics.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset, MetadataEncoder]:
    train_csv = resolve_path(args.train_csv)
    val_csv = resolve_path(args.val_csv)
    test_csv = resolve_path(args.test_csv)
    train_df = pd.read_csv(train_csv)
    metadata_encoder = MetadataEncoder.fit(train_df)

    if args.experiment == "meta_only":
        train_dataset = MetadataOnlyDataset(train_csv, metadata_encoder)
        val_dataset = MetadataOnlyDataset(val_csv, metadata_encoder)
        test_dataset = MetadataOnlyDataset(test_csv, metadata_encoder)
    else:
        train_dataset = HAM10000Dataset(
            train_csv,
            transform=get_train_transform(args.image_size),
            use_metadata=args.experiment == "fusion",
            metadata_encoder=metadata_encoder,
        )
        val_dataset = HAM10000Dataset(
            val_csv,
            transform=get_valid_transform(args.image_size),
            use_metadata=args.experiment == "fusion",
            metadata_encoder=metadata_encoder,
        )
        test_dataset = HAM10000Dataset(
            test_csv,
            transform=get_valid_transform(args.image_size),
            use_metadata=args.experiment == "fusion",
            metadata_encoder=metadata_encoder,
        )
    return train_dataset, val_dataset, test_dataset, metadata_encoder


def build_loaders(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    sampler = None
    if args.weighted_sampler:
        sampler = build_weighted_sampler(resolve_path(args.train_csv))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def unpack_batch(experiment: str, batch, device: torch.device):
    if experiment == "meta_only":
        metadata, labels = batch
        return None, metadata.to(device), labels.to(device)
    if experiment == "image_only":
        images, labels = batch
        return images.to(device), None, labels.to(device)
    images, metadata, labels = batch
    return images.to(device), metadata.to(device), labels.to(device)


def forward_model(model: nn.Module, experiment: str, images, metadata) -> torch.Tensor:
    if experiment == "meta_only":
        return model(metadata)
    if experiment == "image_only":
        return model(images)
    return model(images, metadata)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    y_true = labels.cpu().numpy()
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
    }
    try:
        metrics["macro_auc_ovr"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["macro_auc_ovr"] = float("nan")
    return metrics


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    experiment: str,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    limit_batches: int | None = None,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval", ascii=True)
    for batch_index, batch in enumerate(progress, start=1):
        if limit_batches is not None and batch_index > limit_batches:
            break

        images, metadata, labels = unpack_batch(experiment, batch, device)
        optimizer_context = torch.enable_grad() if is_train else torch.no_grad()
        with optimizer_context:
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = forward_model(model, experiment, images, metadata)
                loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        progress.set_postfix(loss=total_loss / max(total_samples, 1))

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    metrics = compute_metrics(logits_tensor, labels_tensor)
    return total_loss / total_samples, metrics, logits_tensor, labels_tensor


def save_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, output_path: Path) -> None:
    predictions = torch.softmax(logits.float(), dim=1).argmax(dim=1).numpy()
    matrix = confusion_matrix(labels.numpy(), predictions, labels=list(range(len(LABEL_NAMES))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    experiment: str,
    output_dir: Path,
    amp: bool = False,
    limit_batches: int | None = None,
) -> None:
    """Evaluate the best model on the test set and write metrics/artifacts."""
    test_loss, test_metrics, test_logits, test_labels = run_one_epoch(
        model,
        test_loader,
        criterion,
        device,
        experiment,
        amp=amp,
        limit_batches=limit_batches,
    )
    test_report = {"test_loss": test_loss, **test_metrics}
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    save_confusion_matrix(test_logits, test_labels, output_dir / "confusion_matrix.png")
    print(f"test metrics: {test_report}")
    print(f"results saved to: {output_dir.relative_to(PROJECT_ROOT)}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = TrainConfig(
        experiment=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        use_class_weights=not args.no_class_weights,
        use_weighted_sampler=args.weighted_sampler,
        amp=args.amp,
        seed=args.seed,
    )

    output_dir = resolve_path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (output_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    train_dataset, val_dataset, test_dataset, metadata_encoder = build_datasets(args)
    train_loader, val_loader, test_loader = build_loaders(
        args,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    print(f"metadata_dim: {metadata_encoder.num_features}")
    print(f"train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    try:
        model = build_model(
            experiment=args.experiment,
            metadata_dim=metadata_encoder.num_features,
            pretrained=args.pretrained and not args.eval_only,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
    except Exception as exc:
        if args.pretrained:
            raise RuntimeError(
                "加载预训练 EfficientNet-B0 失败。请检查网络，或临时加 --no-pretrained "
                "先跑通流程。"
            ) from exc
        raise

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(resolve_path(args.train_csv)).to(device)
        print(f"class_weights: {[round(x, 4) for x in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = -1.0
    history = []
    best_checkpoint = output_dir / "best_model.pth"

    if args.eval_only:
        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"未找到 checkpoint：{best_checkpoint.relative_to(PROJECT_ROOT)}"
            )
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_and_save(
            model,
            test_loader,
            criterion,
            device,
            args.experiment,
            output_dir,
            amp=args.amp,
            limit_batches=args.limit_test_batches,
        )
        return

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.experiment,
            optimizer=optimizer,
            amp=args.amp,
            limit_batches=args.limit_train_batches,
        )
        val_loss, val_metrics, _, _ = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.experiment,
            amp=args.amp,
            limit_batches=args.limit_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metadata_dim": metadata_encoder.num_features,
                    "sex_to_idx": metadata_encoder.sex_to_idx,
                    "localization_to_idx": metadata_encoder.localization_to_idx,
                    "label_names": LABEL_NAMES,
                    "best_val_macro_f1": best_macro_f1,
                },
                best_checkpoint,
            )
            print(f"saved best checkpoint: {best_checkpoint.relative_to(PROJECT_ROOT)}")

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_and_save(
        model,
        test_loader,
        criterion,
        device,
        args.experiment,
        output_dir,
        amp=args.amp,
        limit_batches=args.limit_test_batches,
    )


if __name__ == "__main__":
    main()
