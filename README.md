# SkinSight - 皮肤病变智能诊断系统

> 数据分析与数据挖掘课程期末项目，选项 C：端到端应用系统开发

本项目基于 HAM10000 数据集构建多模态皮肤病变分类系统，融合皮肤镜图像与患者元数据，完成数据处理、模型训练、评估和交互演示。

## 项目概述

| 项目 | 说明 |
|---|---|
| 数据集 | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| 核心任务 | 识别 7 类皮肤病变 |
| 输入模态 | 皮肤镜图像、年龄、性别、病灶部位 |
| 技术路线 | EfficientNet-B0 迁移学习与元数据特征融合 |
| 实验设计 | 纯图像、纯元数据、图像与元数据融合三组消融实验 |
| 评估指标 | Macro F1、各类别 AUC、混淆矩阵 |
| 演示界面 | Streamlit Web 应用 |

注意：HAM10000 包含 `10015` 张皮肤镜图像。数据类别分布不平衡，其中 `nv` 类样本占比较高，因此模型评估以 **Macro F1** 为主指标，不能只看 Accuracy。

## 项目结构

```text
2026_Tongji_HAM10000/
├── data/
│   ├── raw/                         # 原始数据，不提交到 Git
│   └── processed/                   # 清洗结果与数据集划分 CSV
├── docs/                            # 计划书、技术文档与参考资料
├── notebooks/
│   └── 01_eda_liujie.ipynb          # 探索性数据分析
├── reports/
│   ├── dataset_check_summary.txt    # 数据完整性检查报告
│   └── figures/                     # EDA 图表
├── src/
│   └── data/
│       ├── download_kaggle.py       # 下载并解压 HAM10000
│       ├── check_dataset.py         # 检查原始数据完整性
│       ├── clean_metadata.py        # 清洗元数据并匹配图像路径
│       ├── make_splits.py           # 按 lesion_id 划分数据集
│       └── dataset.py               # PyTorch Dataset 与不平衡处理工具
├── requirements.txt
└── README.md
```

模型、实验配置和 Streamlit 应用将在后续开发中分别加入 `src/models/`、`experiments/` 和 `app/`。

## 当前进度

06.01 数据工程阶段已完成：仓库已包含数据检查、清洗、病灶级划分、EDA 和 PyTorch Dataset 接口。下一阶段将进行模型实现、训练评估和三组消融实验。

## 环境配置

请使用 conda 创建项目专用环境，不要在 `base` 环境中直接安装依赖。

```bash
conda create -n skinsight python=3.10 -y
conda activate skinsight
pip install -r requirements.txt
```

## 数据准备

仓库已经包含 `data/processed/` 下的清洗与划分 CSV。后续模型训练不需要重复执行清洗和划分，只需下载原始图片，并确保图片位于 CSV 中 `image_path` 指向的位置。

### 配置 Kaggle Access Token

下载脚本使用 Kaggle access token。请按照 Kaggle 指引创建 token，并将其保存至：

```text
~/.kaggle/access_token
```

不要将访问令牌提交到仓库。

### 使用已有 CSV 开始训练

执行：

```bash
python src/data/download_kaggle.py
```

下载完成后，图片应位于：

```text
data/raw/ham10000/
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

此时可以直接使用：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

开始模型开发与训练。

### 从头复现数据流水线

仅在首次生成 CSV、修改数据处理逻辑或需要复核数据处理结果时，完整执行：

```bash
python src/data/download_kaggle.py
python src/data/check_dataset.py
python src/data/clean_metadata.py
python src/data/make_splits.py
```

清洗与划分结果位于：

```text
data/processed/
├── metadata_clean.csv
├── train.csv
├── val.csv
├── test.csv
└── split.csv
```

数据集按 `lesion_id` 分组划分，避免同一病灶的不同图像同时进入训练集、验证集和测试集。

### 运行 EDA

```bash
jupyter notebook notebooks/01_eda_liujie.ipynb
```

EDA 图表将保存至 `reports/figures/`。

## 数据接口

`src/data/dataset.py` 提供：

- `HAM10000Dataset`
- `get_train_transform()`
- `get_valid_transform()`
- `compute_class_weights(csv_path)`
- `build_weighted_sampler(csv_path)`

处理后 CSV 的主要字段：

| 字段 | 用途 |
|---|---|
| `image_path` | 相对项目根目录的图片路径 |
| `label` | `0` 到 `6` 的整数类别标签 |
| `dx`、`dx_full_name` | 类别缩写与英文全称 |
| `age`、`sex`、`localization` | 多模态模型可用元数据 |
| `lesion_id` | 病灶级分组字段 |

## 实验设计

| 实验组 | 输入 | 目的 |
|---|---|---|
| `image_only` | 皮肤镜图像 | 图像分类基线 |
| `meta_only` | 年龄、性别、病灶部位 | 验证元数据本身的分类能力 |
| `fusion` | 图像与元数据 | 评估多模态融合效果 |

## 团队分工

| 成员 | 角色 | 主要职责 |
|---|---|---|
| 组员 1 | 数据工程负责人 | 数据获取与检查、元数据清洗、图像预处理、数据集划分、EDA、类别不平衡处理 |
| 组员 2 | 算法与评估负责人 | 模型实现与训练、元数据融合、消融实验、指标计算、结果分析 |
| 组员 3 | 工程与交付负责人 | Streamlit 应用、代码整合、仓库管理、报告、PPT、演示视频 |

## 项目文档

- `docs/plan.md`：技术方案与执行计划
- `docs/数据分析与数据挖掘期末项目计划书.md`：课程项目计划书
- `docs/feature_engineering_liujiye.md`：数据预处理与特征工程说明
- `README_data_liujiye.md`：数据工程运行说明

## 注意事项

- `data/raw/` 和压缩包体积较大，不应提交到 Git。
- HAM10000 类别分布不平衡，训练时应使用类别权重或加权采样。
- 本项目仅用于课程学习与辅助研究展示，不能替代医生诊断。
