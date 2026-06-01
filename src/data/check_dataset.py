"""Check the integrity and basic statistics of the raw HAM10000 dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ham10000"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUMMARY_PATH = REPORTS_DIR / "dataset_check_summary.txt"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [RAW_DIR / "HAM10000_images_part_1", RAW_DIR / "HAM10000_images_part_2"]


def require_file(path: Path) -> None:
    """Raise a readable error when an expected file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到文件：{path.relative_to(PROJECT_ROOT)}。请先运行 python src/data/download_kaggle.py"
        )


def collect_image_paths() -> dict[str, Path]:
    """Collect all jpg images from HAM10000 image folders by image_id."""
    image_paths: dict[str, Path] = {}
    for image_dir in IMAGE_DIRS:
        if not image_dir.exists():
            print(f"警告：未找到图片目录 {image_dir.relative_to(PROJECT_ROOT)}")
            continue
        for path in image_dir.glob("*.jpg"):
            image_paths[path.stem] = path
    return image_paths


def add_line(lines: list[str], text: str = "") -> None:
    """Append one line and print it for console/report parity."""
    lines.append(text)
    print(text)


def main() -> None:
    """Run dataset integrity checks and write a text summary."""
    require_file(METADATA_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    image_paths = collect_image_paths()

    missing_image_mask = ~df["image_id"].isin(image_paths)
    duplicate_image_id_count = int(df.duplicated("image_id").sum())
    duplicate_lesion_id_count = int(df.duplicated("lesion_id").sum())

    lines: list[str] = []
    add_line(lines, "HAM10000 数据完整性检查结果")
    add_line(lines, "=" * 40)
    add_line(lines, f"metadata shape: {df.shape}")
    add_line(lines, f"字段名: {list(df.columns)}")
    add_line(lines, "\n前 5 行:")
    add_line(lines, df.head().to_string(index=False))

    add_line(lines, "\n图片数量检查:")
    for image_dir in IMAGE_DIRS:
        jpg_count = len(list(image_dir.glob("*.jpg"))) if image_dir.exists() else 0
        add_line(lines, f"- {image_dir.relative_to(PROJECT_ROOT)}: {jpg_count} 张 jpg")
    add_line(lines, f"- 合计可索引图片数量: {len(image_paths)}")

    add_line(lines, "\n完整性检查:")
    add_line(lines, f"metadata 中无法找到对应图片的记录数: {int(missing_image_mask.sum())}")
    add_line(lines, f"重复 image_id 数量: {duplicate_image_id_count}")
    add_line(lines, f"重复 lesion_id 数量: {duplicate_lesion_id_count}")

    add_line(lines, "\ndx 类别分布:")
    add_line(lines, df["dx"].value_counts().to_string())

    add_line(lines, "\n关键字段缺失值情况:")
    for col in ["age", "sex", "localization", "dx_type"]:
        missing_count = int(df[col].isna().sum()) if col in df.columns else -1
        unknown_count = int((df[col].astype(str).str.lower() == "unknown").sum()) if col in df.columns else -1
        add_line(lines, f"- {col}: missing={missing_count}, unknown={unknown_count}")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n检查结果已保存到：{SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
