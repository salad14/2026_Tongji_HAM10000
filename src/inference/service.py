"""Application-facing inference service."""

from .schema import ModelVariant, PatientMetadata, PredictionResult, Predictor


_DEFAULT_PREDICTOR: Predictor | None = None


class PredictorNotConfiguredError(ValueError):
    """Raised when inference is called before a real predictor is installed."""


def predict(
    image: bytes | None,
    metadata: PatientMetadata,
    variant: ModelVariant = "fusion",
    predictor: Predictor | None = None,
) -> PredictionResult:
    """Predict one sample through the configured provider."""
    active_predictor = predictor if predictor is not None else _DEFAULT_PREDICTOR
    if active_predictor is None:
        raise PredictorNotConfiguredError("真实模型尚未接入：请先配置 PyTorch 推理适配器。")
    return active_predictor.predict(image=image, metadata=metadata, variant=variant)
