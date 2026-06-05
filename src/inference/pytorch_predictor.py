"""PyTorch predictor adapter for trained HAM10000 checkpoints."""

from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
from PIL import Image

from src.data.dataset import MetadataEncoder, get_valid_transform
from src.models import build_model

from .schema import CLASS_LABELS, MODEL_VARIANTS, ModelVariant, PatientMetadata, PredictionResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def select_device() -> torch.device:
    """Use GPU by default when the current PyTorch build can access CUDA."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a checkpoint using PyTorch's safer weights-only mode."""
    if not path.exists():
        raise FileNotFoundError(f"未找到模型权重：{path.relative_to(PROJECT_ROOT)}")
    return torch.load(path, map_location=device, weights_only=True)


def _metadata_encoder_from_checkpoint(checkpoint: dict) -> MetadataEncoder:
    """Restore metadata encoders saved by the training pipeline."""
    encoder = MetadataEncoder([], [])
    encoder.sex_to_idx = dict(checkpoint["sex_to_idx"])
    encoder.localization_to_idx = dict(checkpoint["localization_to_idx"])
    return encoder


class PytorchPredictor:
    """Run one trained HAM10000 model variant."""

    def __init__(
        self,
        variant: ModelVariant,
        checkpoint_path: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"不支持的模型类型：{variant}")

        self.variant = variant
        self.device = device or select_device()
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else OUTPUTS_DIR / variant / "best_model.pth"
        )
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = PROJECT_ROOT / self.checkpoint_path

        checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        label_names = tuple(checkpoint.get("label_names", CLASS_LABELS))
        if label_names != CLASS_LABELS:
            raise ValueError(f"模型类别顺序不符合应用契约：{label_names}")

        self.encoder = _metadata_encoder_from_checkpoint(checkpoint)
        image_size = checkpoint.get("config", {}).get("image_size", 224)
        self.image_size = int(image_size)
        self.transform = get_valid_transform(self.image_size)
        self.model = build_model(
            variant,
            metadata_dim=self.encoder.num_features,
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    @property
    def provider(self) -> str:
        """Return a short provider label for the UI."""
        return f"pytorch-{self.device.type}"

    def predict(
        self,
        image: bytes | None,
        metadata: PatientMetadata,
        variant: ModelVariant,
    ) -> PredictionResult:
        """Predict one sample and return normalized probabilities."""
        if variant != self.variant:
            raise ValueError(f"当前 predictor 为 {self.variant}，不能用于 {variant}。")
        if variant in {"image_only", "fusion"} and not image:
            raise ValueError("当前模型类型需要上传皮肤镜图片。")

        metadata_tensor = self._metadata_tensor(metadata)
        image_tensor = self._image_tensor(image) if image is not None else None

        with torch.inference_mode():
            if variant == "meta_only":
                logits = self.model(metadata_tensor)
            elif variant == "image_only":
                logits = self.model(image_tensor)
            else:
                logits = self.model(image_tensor, metadata_tensor)
            raw_probabilities = torch.softmax(logits.float(), dim=1).squeeze(0).detach().cpu().tolist()

        total = sum(raw_probabilities)
        probabilities = {
            label: float(probability / total)
            for label, probability in zip(CLASS_LABELS, raw_probabilities)
        }
        predicted_class = max(probabilities, key=probabilities.get)
        return PredictionResult(
            variant=variant,
            probabilities=probabilities,
            predicted_class=predicted_class,
            provider=self.provider,
        )

    def _metadata_tensor(self, metadata: PatientMetadata) -> torch.Tensor:
        tensor = self.encoder.encode(
            {
                "age": metadata.age,
                "sex": metadata.sex,
                "localization": metadata.localization,
            }
        )
        return tensor.unsqueeze(0).to(self.device)

    def _image_tensor(self, image: bytes | None) -> torch.Tensor:
        if image is None:
            raise ValueError("当前模型类型需要上传皮肤镜图片。")
        try:
            pil_image = Image.open(BytesIO(image)).convert("RGB")
        except Exception as exc:
            raise ValueError("无法读取上传图片，请确认文件为有效的 JPG 或 PNG。") from exc
        array = np.asarray(pil_image)
        return self.transform(image=array)["image"].unsqueeze(0).to(self.device)


@lru_cache(maxsize=len(MODEL_VARIANTS))
def get_predictor(variant: ModelVariant) -> PytorchPredictor:
    """Cache trained predictors for the Streamlit process."""
    return PytorchPredictor(variant)
