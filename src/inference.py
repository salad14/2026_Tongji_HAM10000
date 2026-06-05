"""Inference helpers for the trained HAM10000 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MetadataEncoder, get_valid_transform  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import LABEL_NAMES  # noqa: E402


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    """Load a training checkpoint from disk."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到模型权重：{checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def metadata_encoder_from_checkpoint(checkpoint: dict) -> MetadataEncoder:
    """Restore the metadata encoder maps saved during training."""
    encoder = MetadataEncoder([], [])
    encoder.sex_to_idx = dict(checkpoint["sex_to_idx"])
    encoder.localization_to_idx = dict(checkpoint["localization_to_idx"])
    return encoder


def load_model(
    checkpoint_path: str | Path,
    experiment: str = "fusion",
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, MetadataEncoder, torch.device]:
    """Load a trained model and its metadata encoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    encoder = metadata_encoder_from_checkpoint(checkpoint)
    model = build_model(
        experiment=experiment,
        metadata_dim=encoder.num_features,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, encoder, device


def load_image_tensor(image_path: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    """Load one image and return a batched tensor."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_valid_transform(image_size)
    tensor = transform(image=image)["image"].unsqueeze(0)
    return tensor.to(device)


def predict_one(
    model: torch.nn.Module,
    encoder: MetadataEncoder,
    device: torch.device,
    image_path: str | Path,
    age: float,
    sex: str,
    localization: str,
    experiment: str = "fusion",
    image_size: int = 224,
) -> dict[str, float]:
    """Predict class probabilities for one sample."""
    image = load_image_tensor(image_path, image_size, device)
    metadata = encoder.encode(
        {
            "age": age,
            "sex": sex,
            "localization": localization,
        }
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        if experiment == "image_only":
            logits = model(image)
        elif experiment == "meta_only":
            logits = model(metadata)
        else:
            logits = model(image, metadata)
        probabilities = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().tolist()
    return {label: float(probability) for label, probability in zip(LABEL_NAMES, probabilities)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one HAM10000 prediction.")
    parser.add_argument("--checkpoint", default="outputs/fusion/best_model.pth")
    parser.add_argument("--experiment", choices=["meta_only", "image_only", "fusion"], default="fusion")
    parser.add_argument("--image", required=True)
    parser.add_argument("--age", type=float, default=50.0)
    parser.add_argument("--sex", default="unknown")
    parser.add_argument("--localization", default="unknown")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, encoder, device = load_model(args.checkpoint, args.experiment)
    probabilities = predict_one(
        model=model,
        encoder=encoder,
        device=device,
        image_path=args.image,
        age=args.age,
        sex=args.sex,
        localization=args.localization,
        experiment=args.experiment,
        image_size=args.image_size,
    )
    prediction = max(probabilities, key=probabilities.get)
    print(
        json.dumps(
            {
                "prediction": prediction,
                "probabilities": probabilities,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
