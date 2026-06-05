"""Inference interfaces for the SkinSight application."""

from .pytorch_predictor import PytorchPredictor, select_device
from .schema import PatientMetadata, PredictionResult
from .service import PredictorNotConfiguredError, predict

__all__ = [
    "PatientMetadata",
    "PredictionResult",
    "PredictorNotConfiguredError",
    "PytorchPredictor",
    "predict",
    "select_device",
]
