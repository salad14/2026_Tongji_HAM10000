"""Create lesion-level train/validation/test splits for HAM10000."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "metadata_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
SPLIT_PATH = PROCESSED_DIR / "split.csv"
RANDOM_STATE = 42


def require_file(path: Path) -> None:
    """Raise a readable error when an expected file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到文件：{path.relative_to(PROJECT_ROOT)}。请先运行 python src/data/clean_metadata.py"
        )


def group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by lesion_id to avoid lesion-level leakage."""
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(first_split.split(df, groups=df["lesion_id"]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    second_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(second_split.split(temp_df, groups=temp_df["lesion_id"]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    return train_df, val_df, test_df


def check_no_group_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Ensure lesion_id values do not overlap across splits."""
    train_groups = set(train_df["lesion_id"])
    val_groups = set(val_df["lesion_id"])
    test_groups = set(test_df["lesion_id"])

    overlaps = {
        "train-val": train_groups & val_groups,
        "train-test": train_groups & test_groups,
        "val-test": val_groups & test_groups,
    }
    bad_overlaps = {name: groups for name, groups in overlaps.items() if groups}
    if bad_overlaps:
        raise ValueError(f"发现 lesion_id 交集，存在数据泄漏风险：{bad_overlaps}")


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    """Print sample count and class distribution for one split."""
    print(f"\n{name} 样本数量: {len(df)}")
    print("类别分布:")
    print(df["dx"].value_counts().sort_index().to_string())


def main() -> None:
    """Create train.csv, val.csv, test.csv and split.csv."""
    require_file(INPUT_PATH)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    train_df, val_df, test_df = group_split(df)
    check_no_group_overlap(train_df, val_df, test_df)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df.to_csv(SPLIT_PATH, index=False, encoding="utf-8")

    print_split_summary("train", train_df)
    print_split_summary("val", val_df)
    print_split_summary("test", test_df)
    print("\n已确认 train、val、test 之间无 lesion_id 重叠。")
    print(f"已保存：{TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"已保存：{VAL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"已保存：{TEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"已保存：{SPLIT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
