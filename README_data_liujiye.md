# HAM10000 数据工程说明（刘继业 2252752）

本部分负责“基于多模态数据的皮肤病变智能诊断系统”的数据工程流程：从 Kaggle 下载 HAM10000 数据集，完成数据完整性检查、元数据清洗、病灶级数据集划分、EDA 图表生成，并提供可供算法同学直接训练使用的 PyTorch Dataset 接口。

## 1. macOS 环境配置

推荐使用 conda 创建独立环境：

```bash
conda create -n ham10000 python=3.10 -y
conda activate ham10000
pip install -r requirements.txt
```

## 2. 配置 Kaggle API

本项目下载脚本使用 Kaggle access token。

```bash
mkdir -p ~/.kaggle
echo <你的 KGAT access token> > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

`access_token` 不要提交到 GitHub。

## 3. 下载数据

```bash
python src/data/download_kaggle.py
```

下载目标路径：

```text
data/raw/ham10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

## 4. 检查数据

```bash
python src/data/check_dataset.py
```

输出检查结果：

```text
reports/dataset_check_summary.txt
```

## 5. 清洗 metadata

```bash
python src/data/clean_metadata.py
```

输出：

```text
data/processed/metadata_clean.csv
```

## 6. 划分数据集

```bash
python src/data/make_splits.py
```

输出：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/split.csv
```

划分使用 `lesion_id` 做 group split，避免同一病灶的多张图片同时出现在训练集、验证集和测试集中。

## 7. 测试 Dataset

```bash
python src/data/dataset.py
```

该脚本会读取 `train.csv`，加载一个 batch，并打印 `image shape`、`metadata shape` 和 `label shape`。

## 8. 运行 EDA Notebook

```bash
jupyter notebook notebooks/01_eda_liujiye.ipynb
```

Notebook 会生成以下图表：

```text
reports/figures/class_distribution.png
reports/figures/age_distribution.png
reports/figures/sex_distribution.png
reports/figures/localization_distribution.png
reports/figures/dx_type_distribution.png
reports/figures/missing_values.png
reports/figures/sample_images_by_class.png
```

## 9. 不应提交到 GitHub 的文件

以下内容已在 `.gitignore` 中忽略：

- `data/raw/`
- `data/processed/`
- `*.zip`
- `__pycache__/`
- `.DS_Store`
- `.ipynb_checkpoints/`
- `kaggle.json`
- `access_token`
- `access_token.txt`

## 10. 交付给算法同学的文件和接口

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `image_path` 字段：图片路径
- `label` 字段：整数类别标签
- `src/data/dataset.py` 中的 `HAM10000Dataset`
- `get_train_transform()`
- `get_valid_transform()`
- `compute_class_weights(csv_path)`
- `build_weighted_sampler(csv_path)`
