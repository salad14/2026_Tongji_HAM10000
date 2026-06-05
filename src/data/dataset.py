"""PyTorch Dataset, transforms and imbalance utilities for HAM10000."""

from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MetadataEncoder:
    """Encode age, sex and localization into a stable numeric feature vector.

    Fit this encoder on the training CSV, then reuse it for validation and test
    datasets. This keeps category indices consistent across all splits.
    """

    def __init__(self, sex_values: Iterable[str], localization_values: Iterable[str]) -> None:
        self.sex_to_idx = self._build_category_map(sex_values)
        self.localization_to_idx = self._build_category_map(localization_values)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "MetadataEncoder":
        """Create an encoder from a training dataframe."""
        return cls(df["sex"], df["localization"])

    @staticmethod
    def _build_category_map(values: Iterable[str]) -> dict[str, int]:
        normalized = pd.Series(values).fillna("unknown").astype(str).str.lower().unique()
        sorted_values = sorted(normalized)
        if "unknown" in sorted_values:
            sorted_values.remove("unknown")
        sorted_values = ["unknown"] + sorted_values
        return {value: idx for idx, value in enumerate(sorted_values)}

    @property
    def num_features(self) -> int:
        """Return metadata feature dimension: age + sex one-hot + location one-hot."""
        return 1 + len(self.sex_to_idx) + len(self.localization_to_idx)

    def encode(self, row: pd.Series) -> torch.Tensor:
        """Return a tensor containing normalized age and one-hot categorical features."""
        age = float(row["age"]) / 100.0
        features = torch.zeros(self.num_features, dtype=torch.float32)
        features[0] = age

        sex = str(row["sex"]).lower()
        sex_idx = self.sex_to_idx.get(sex, self.sex_to_idx["unknown"])
        features[1 + sex_idx] = 1.0

        localization = str(row["localization"]).lower()
        loc_idx = self.localization_to_idx.get(localization, self.localization_to_idx["unknown"])
        loc_offset = 1 + len(self.sex_to_idx)
        features[loc_offset + loc_idx] = 1.0
        return features


def get_train_transform(image_size: int = 224) -> A.Compose:
    """Return image transforms with random augmentation for training."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_valid_transform(image_size: int = 224) -> A.Compose:
    """Return deterministic image transforms for validation and testing."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class HAM10000Dataset(Dataset):
    """HAM10000 dataset that can return image-only or image+metadata samples."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: A.Compose | None = None,
        use_metadata: bool = False,
        metadata_encoder: MetadataEncoder | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        if not self.csv_path.exists():
            raise FileNotFoundError(f"未找到 CSV 文件：{self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        required_columns = {"image_path", "label", "age", "sex", "localization"}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"CSV 缺少必要字段：{sorted(missing_columns)}")

        self.transform = transform if transform is not None else get_valid_transform()
        self.use_metadata = use_metadata
        self.metadata_encoder = metadata_encoder or MetadataEncoder.fit(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve project-relative image paths saved in metadata_clean.csv."""
        path = Path(image_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Read an image with OpenCV and convert BGR to RGB."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _metadata_tensor(self, row: pd.Series) -> torch.Tensor:
        """Create a metadata feature tensor using the shared encoder."""
        return self.metadata_encoder.encode(row)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(row["image_path"])
        image = self._load_image(image_path)
        image = self.transform(image=image)["image"]
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        if self.use_metadata:
            return image, self._metadata_tensor(row), label
        return image, label


def compute_class_weights(csv_path: str | Path) -> torch.Tensor:
    """Compute balanced class weights from a training CSV."""
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(int).to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(csv_path: str | Path) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that samples minority classes more often."""
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(int)
    class_counts = labels.value_counts().to_dict()
    sample_weights = labels.map(lambda label: 1.0 / class_counts[int(label)]).to_numpy()
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main() -> None:
    """Smoke-test the Dataset by loading one mini-batch from train.csv."""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"未找到 {TRAIN_CSV.relative_to(PROJECT_ROOT)}。请先运行 python src/data/make_splits.py"
        )

    dataset = HAM10000Dataset(
        TRAIN_CSV,
        transform=get_train_transform(),
        use_metadata=True,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, metadata, labels = next(iter(loader))

    print(f"image shape: {tuple(images.shape)}")
    print(f"metadata shape: {tuple(metadata.shape)}")
    print(f"label shape: {tuple(labels.shape)}")
    print(f"metadata feature dim: {dataset.metadata_encoder.num_features}")
    print(f"class weights: {compute_class_weights(TRAIN_CSV).tolist()}")


if __name__ == "__main__":
    main()
