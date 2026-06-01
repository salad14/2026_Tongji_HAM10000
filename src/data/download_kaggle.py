"""Download and unzip the HAM10000 dataset with a Kaggle access token.

This script uses the new Kaggle access token file:
~/.kaggle/access_token

It deliberately avoids the legacy KaggleApi Python SDK path because some
environments still force ~/.kaggle/kaggle.json during SDK import.
"""

from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_OWNER = "kmader"
DATASET_SLUG = "skin-cancer-mnist-ham10000"
DATASET = f"{DATASET_OWNER}/{DATASET_SLUG}"
DOWNLOAD_API_URL = "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ham10000"
DATASET_ZIP = RAW_DIR / f"{DATASET_SLUG}.zip"
KAGGLE_ACCESS_TOKEN = Path.home() / ".kaggle" / "access_token"


def print_kaggle_help() -> None:
    """Print a clear setup message for users who have not configured Kaggle."""
    print("未检测到 Kaggle access token。请按下面方式配置：")
    print("  mkdir -p ~/.kaggle")
    print("  echo <你的 KGAT access token> > ~/.kaggle/access_token")
    print("  chmod 600 ~/.kaggle/access_token")


def read_access_token() -> str | None:
    """Read the Kaggle access token from ~/.kaggle/access_token."""
    if not KAGGLE_ACCESS_TOKEN.exists():
        return None

    token = KAGGLE_ACCESS_TOKEN.read_text(encoding="utf-8").strip()
    if not token:
        print("~/.kaggle/access_token 文件为空，请重新写入 KGAT access token。")
        return None
    return token


def check_key_files() -> None:
    """Print whether the expected HAM10000 files are present after download."""
    expected_paths = [
        RAW_DIR / "HAM10000_metadata.csv",
        RAW_DIR / "HAM10000_images_part_1",
        RAW_DIR / "HAM10000_images_part_2",
    ]
    print("\n下载后关键文件检查：")
    for path in expected_paths:
        status = "存在" if path.exists() else "缺失"
        print(f"- {path.relative_to(PROJECT_ROOT)}: {status}")


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip file while preventing path traversal."""
    target_dir = target_dir.resolve()
    with ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            output_path = (target_dir / member.filename).resolve()
            if not str(output_path).startswith(str(target_dir)):
                raise ValueError(f"zip 内存在不安全路径：{member.filename}")
        zip_file.extractall(target_dir)


def unzip_dataset_archive() -> None:
    """Unzip the main dataset archive into data/raw/ham10000."""
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(f"下载 zip 不存在：{DATASET_ZIP.relative_to(PROJECT_ROOT)}")

    print(f"正在解压 {DATASET_ZIP.name} 到 {RAW_DIR.relative_to(PROJECT_ROOT)}")
    safe_extract_zip(DATASET_ZIP, RAW_DIR)


def unzip_nested_image_archives() -> None:
    """Unzip image archives that are bundled inside the Kaggle dataset zip."""
    for zip_path in [
        RAW_DIR / "HAM10000_images_part_1.zip",
        RAW_DIR / "HAM10000_images_part_2.zip",
    ]:
        target_dir = RAW_DIR / zip_path.stem
        if target_dir.exists() and any(target_dir.glob("*.jpg")):
            continue
        if not zip_path.exists():
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"正在解压 {zip_path.name} 到 {target_dir.relative_to(PROJECT_ROOT)}")
        with ZipFile(zip_path, "r") as zip_file:
            for member in zip_file.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".jpg"):
                    continue
                output_path = target_dir / Path(member.filename).name
                with zip_file.open(member) as source, output_path.open("wb") as target:
                    target.write(source.read())


def download_with_access_token(token: str) -> None:
    """Download the Kaggle dataset zip using Bearer token authentication."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-api/v1.7.0",
    }
    payload = {
        "ownerSlug": DATASET_OWNER,
        "datasetSlug": DATASET_SLUG,
    }

    print(f"使用 Kaggle access token 下载数据集：{DATASET}")
    print(f"目标文件：{DATASET_ZIP.relative_to(PROJECT_ROOT)}")

    try:
        response = requests.post(
            DOWNLOAD_API_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Kaggle 下载请求失败，请检查网络连接：{exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(f"Kaggle 返回了非预期响应：{response.text}") from None
        if "url" in payload:
            download_from_signed_url(payload["url"])
            return
        raise RuntimeError(f"Kaggle 返回了 JSON 响应，下载未成功：{payload}")

    if response.status_code in {401, 403}:
        raise RuntimeError("Kaggle access token 无效、过期，或没有访问该数据集的权限。")
    response.raise_for_status()
    stream_response_to_file(response)


def download_from_signed_url(url: str) -> None:
    """Download the dataset zip from a signed Kaggle redirect URL."""
    try:
        response = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Kaggle 签名下载地址请求失败：{exc}") from exc

    response.raise_for_status()
    stream_response_to_file(response)


def stream_response_to_file(response: requests.Response) -> None:
    """Write a streaming HTTP response to the dataset zip file."""
    total_size = int(response.headers.get("Content-Length", 0))
    with DATASET_ZIP.open("wb") as output_file:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=DATASET_ZIP.name) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output_file.write(chunk)
                progress.update(len(chunk))


def main() -> None:
    """Download and unzip the Kaggle dataset."""
    token = read_access_token()
    if token is None:
        print_kaggle_help()
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("已检测到 Kaggle access token：~/.kaggle/access_token")

    try:
        download_with_access_token(token)
        unzip_dataset_archive()
        unzip_nested_image_archives()
    except Exception as exc:
        print("Kaggle 数据集下载或解压失败。")
        print(f"错误信息：{exc}")
        return

    check_key_files()


if __name__ == "__main__":
    main()
