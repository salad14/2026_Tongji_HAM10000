"""Shared inference data structures for application predictors."""

import math
from dataclasses import dataclass
from typing import Literal, Protocol


ModelVariant = Literal["image_only", "meta_only", "fusion"]
CLASS_LABELS = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MODEL_VARIANTS = ("image_only", "meta_only", "fusion")
SEX_OPTIONS = ("female", "male", "unknown")
LOCALIZATION_OPTIONS = (
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "unknown",
    "upper extremity",
)


@dataclass(frozen=True)
class PatientMetadata:
    """Metadata collected by the diagnosis form."""

    age: float
    sex: str
    localization: str

    def __post_init__(self) -> None:
        """Validate user-provided metadata."""
        if not math.isfinite(self.age) or not 0 <= self.age <= 120:
            raise ValueError("年龄必须在 0 到 120 之间。")
        if self.sex not in SEX_OPTIONS:
            raise ValueError(f"不支持的性别编码：{self.sex}")
        if not self.localization:
            raise ValueError("病灶部位不能为空。")


@dataclass(frozen=True)
class PredictionResult:
    """Normalized output returned by every predictor implementation."""

    variant: ModelVariant
    probabilities: dict[str, float]
    predicted_class: str
    provider: str
    warning: str | None = None

    def __post_init__(self) -> None:
        """Validate predictor output before it reaches the UI."""
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(f"不支持的模型类型：{self.variant}")
        if set(self.probabilities) != set(CLASS_LABELS):
            raise ValueError("预测结果必须包含全部 7 个类别。")
        if any(not math.isfinite(value) or value < 0 for value in self.probabilities.values()):
            raise ValueError("类别概率不能为负数。")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("类别概率之和必须为 1。")
        if self.predicted_class not in CLASS_LABELS:
            raise ValueError(f"未知预测类别：{self.predicted_class}")


class Predictor(Protocol):
    """Interface implemented by concrete predictor adapters."""

    def predict(
        self,
        image: bytes | None,
        metadata: PatientMetadata,
        variant: ModelVariant,
    ) -> PredictionResult:
        """Predict one sample and return normalized probabilities."""
