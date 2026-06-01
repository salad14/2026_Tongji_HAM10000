"""Clean HAM10000 metadata and attach image paths for model training."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ham10000"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.csv"
OUTPUT_PATH = PROCESSED_DIR / "metadata_clean.csv"
IMAGE_DIRS = [RAW_DIR / "HAM10000_images_part_1", RAW_DIR / "HAM10000_images_part_2"]

LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

DX_FULL_NAME_MAP = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def require_file(path: Path) -> None:
    """Raise a readable error when an expected file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到文件：{path.relative_to(PROJECT_ROOT)}。请先运行 python src/data/download_kaggle.py"
        )


def build_image_path_map() -> dict[str, str]:
    """Map image_id to a project-relative image path."""
    image_path_map: dict[str, str] = {}
    for image_dir in IMAGE_DIRS:
        if not image_dir.exists():
            print(f"警告：未找到图片目录 {image_dir.relative_to(PROJECT_ROOT)}")
            continue
        for image_path in image_dir.glob("*.jpg"):
            relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()
            image_path_map[image_path.stem] = relative_path
    return image_path_map


def fill_unknown(series: pd.Series) -> pd.Series:
    """Normalize missing values and 'unknown' category values to unknown."""
    cleaned = series.fillna("unknown").astype(str).str.strip().str.lower()
    cleaned = cleaned.replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    return cleaned


def main() -> None:
    """Clean metadata and save data/processed/metadata_clean.csv."""
    require_file(METADATA_PATH)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    before_count = len(df)
    image_path_map = build_image_path_map()

    df["image_path"] = df["image_id"].map(image_path_map)
    df = df.dropna(subset=["image_path"]).copy()
    df = df.drop_duplicates(subset=["image_id"], keep="first").copy()

    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)
    df["sex"] = fill_unknown(df["sex"])
    df["localization"] = fill_unknown(df["localization"])

    df["label"] = df["dx"].map(LABEL_MAP)
    df["dx_full_name"] = df["dx"].map(DX_FULL_NAME_MAP)

    if df["label"].isna().any():
        unknown_labels = sorted(df.loc[df["label"].isna(), "dx"].unique())
        raise ValueError(f"发现未定义的 dx 标签：{unknown_labels}")

    df["label"] = df["label"].astype(int)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"清洗前样本数量: {before_count}")
    print(f"清洗后样本数量: {len(df)}")
    print("\n各类别数量:")
    print(df["dx"].value_counts().sort_index().to_string())
    print(f"\n已保存：{OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
