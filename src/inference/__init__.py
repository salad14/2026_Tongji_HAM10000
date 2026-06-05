"""Inference interfaces for the SkinSight application."""

from .schema import PatientMetadata, PredictionResult
from .service import PredictorNotConfiguredError, predict

__all__ = ["PatientMetadata", "PredictionResult", "PredictorNotConfiguredError", "predict"]
