"""Application-facing inference service."""

from .pytorch_predictor import get_predictor
from .schema import ModelVariant, PatientMetadata, PredictionResult, Predictor


class PredictorNotConfiguredError(ValueError):
    """Raised when inference is called before model weights are available."""


def predict(
    image: bytes | None,
    metadata: PatientMetadata,
    variant: ModelVariant = "fusion",
    predictor: Predictor | None = None,
) -> PredictionResult:
    """Predict one sample through the configured provider."""
    try:
        active_predictor = predictor if predictor is not None else get_predictor(variant)
    except FileNotFoundError as exc:
        raise PredictorNotConfiguredError("未找到真实模型权重，请确认 outputs/ 目录已放在项目根目录。") from exc
    return active_predictor.predict(image=image, metadata=metadata, variant=variant)
