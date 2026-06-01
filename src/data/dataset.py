"""PyTorch Dataset, transforms and imbalance utilities for HAM10000."""

from pathlib import Path

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
        self.sex_to_idx = self._build_category_map("sex")
        self.localization_to_idx = self._build_category_map("localization")

    def _build_category_map(self, column: str) -> dict[str, int]:
        """Build a deterministic category-to-index mapping from a CSV column."""
        values = self.df[column].fillna("unknown").astype(str).str.lower().unique()
        sorted_values = sorted(values)
        if "unknown" in sorted_values:
            sorted_values.remove("unknown")
            sorted_values = ["unknown"] + sorted_values
        return {value: idx for idx, value in enumerate(sorted_values)}

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
        """Create a compact metadata tensor: age, sex id and localization id."""
        age = float(row["age"]) / 100.0
        sex = str(row["sex"]).lower()
        localization = str(row["localization"]).lower()
        sex_idx = self.sex_to_idx.get(sex, self.sex_to_idx.get("unknown", 0))
        localization_idx = self.localization_to_idx.get(
            localization, self.localization_to_idx.get("unknown", 0)
        )
        return torch.tensor([age, float(sex_idx), float(localization_idx)], dtype=torch.float32)

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
    print(f"class weights: {compute_class_weights(TRAIN_CSV).tolist()}")


if __name__ == "__main__":
    main()
