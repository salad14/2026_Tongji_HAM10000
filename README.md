# SkinSight - 皮肤病变智能诊断系统

> 数据分析与数据挖掘课程期末项目，选项 C：端到端应用系统开发。

SkinSight 基于 HAM10000 数据集构建多模态皮肤病变七分类系统，融合皮肤镜图像与患者元数据，覆盖数据处理、模型训练、测试集评估、消融实验和 Streamlit 交互展示。

## 项目概览

| 项目 | 说明 |
|---|---|
| 数据集 | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| 核心任务 | 识别 7 类皮肤病变：`akiec`、`bcc`、`bkl`、`df`、`mel`、`nv`、`vasc` |
| 输入模态 | 皮肤镜图像、年龄、性别、病灶部位 |
| 技术路线 | EfficientNet-B0 图像分支、元数据 MLP 分支、多模态融合分支 |
| 实验设计 | `image_only`、`meta_only`、`fusion` 三组消融实验 |
| 主要指标 | Macro F1、Accuracy、AUC、混淆矩阵 |
| 展示界面 | Streamlit Web 应用 |

HAM10000 共包含 `10015` 张皮肤镜图像，类别分布明显不均衡，其中 `nv` 类样本占比较高。因此模型分析以 **Macro F1** 为重点，不只依赖 Accuracy。

## 项目结构

```text
2026_Tongji_HAM10000/
|-- app/                            # Streamlit 应用
|   |-- main.py                     # 首页
|   `-- pages/
|       |-- 1_diagnosis.py          # 单病例诊断
|       |-- 2_model_evaluation.py   # 模型评估结果
|       |-- 3_ablation_study.py     # 三分支消融对比
|       `-- 4_eda.py                # 数据探索
|-- data/
|   |-- raw/                        # 原始图片，不提交到 Git
|   `-- processed/                  # 清洗结果与数据集划分 CSV
|-- docs/                           # 计划书、技术文档与交接说明
|-- notebooks/
|   `-- 01_eda_liujie.ipynb         # 探索性数据分析
|-- outputs/                        # 模型权重与训练输出，不提交到 Git
|-- reports/
|   |-- 项目报告.md / 项目报告.pdf   # 最终项目报告
|   `-- figures/                    # EDA 图表
|-- src/
|   |-- data/                       # 下载、清洗、划分与 Dataset
|   |-- inference/                  # 应用侧推理接口与 PyTorch 适配器
|   |-- models/                     # HAM10000 三分支模型
|   `-- train.py                    # 训练、评估与重新评估脚本
|-- tests/                          # 推理接口测试
|-- requirements.txt
`-- README.md
```

模型权重和训练输出位于 `outputs/`，该目录不提交到 Git。当前应用默认从以下路径读取模型：

```text
outputs/meta_only/best_model.pth
outputs/image_only/best_model.pth
outputs/fusion/best_model.pth
```

### 模型权重下载

三个训练好的模型权重体积较大、未提交到 Git，已通过百度网盘单独提供：

| 项目 | 内容 |
|---|---|
| 链接 | <https://pan.baidu.com/s/1nIE4oXWtUdpxejEIm9Iisg?pwd=6cw2> |
| 提取码 | `6cw2` |
| 文件 | `model.zip` |

下载并解压后，请确保三个权重文件最终位于上述路径，即项目根目录下形成如下结构：

```text
outputs/
|-- meta_only/best_model.pth
|-- image_only/best_model.pth
`-- fusion/best_model.pth
```

放置到位后即可直接 `streamlit run app/main.py` 进行真实推理，无需重新训练。

## 当前状态

项目已完成数据处理、三分支模型训练、测试集评估和 Streamlit 应用接入。应用当前使用真实 PyTorch 模型推理；当 CUDA 可用时默认使用 GPU，CPU 仍可作为兼容回退。模型文件不进入仓库，需要按上文「模型权重下载」从网盘获取并放入 `outputs/`。

## 环境配置

建议使用项目专用 conda 环境：

```bash
conda create -n skinsight python=3.10 -y
conda activate skinsight
pip install -r requirements.txt
```

如需使用 GPU 推理或训练，请安装与本机 CUDA 驱动匹配的 PyTorch CUDA 版本。CPU 推理仍可运行，但图像分支和融合分支响应速度会慢一些。

## 数据准备

仓库已包含 `data/processed/` 下的清洗与划分 CSV。若只是运行应用中的 EDA 页面或使用已训练模型推理，不需要重新执行数据清洗和划分。

训练模型时需要下载原始图片，并确保图片位于 CSV 中 `image_path` 指向的位置：

```bash
python src/data/download_kaggle.py
```

图片下载完成后，目录通常为：

```text
data/raw/ham10000/
|-- HAM10000_images_part_1/
`-- HAM10000_images_part_2/
```

仅在首次生成 CSV、修改数据处理逻辑或需要复核数据处理结果时，才完整执行：

```bash
python src/data/download_kaggle.py
python src/data/check_dataset.py
python src/data/clean_metadata.py
python src/data/make_splits.py
```

## 运行应用

```bash
streamlit run app/main.py
```

应用包含四个功能页：

| 页面 | 功能 |
|---|---|
| 病变诊断 | 上传图片并填写元数据，输出七分类概率和最高概率类别 |
| 模型评估 | 读取 `outputs/*/test_metrics.json` 和混淆矩阵，展示测试集结果 |
| 消融实验 | 对同一病例运行三种模型分支，比较模态贡献 |
| 数据探索 | 读取现有 CSV 与 EDA 图表，展示类别、年龄、性别和部位分布 |

本项目仅用于课程学习与辅助研究展示，不能替代医生诊断。

## 模型训练与评估

训练入口：

```bash
python src/train.py --experiment meta_only --epochs 30 --batch-size 128
python src/train.py --experiment image_only --epochs 30 --batch-size 32 --amp
python src/train.py --experiment fusion --epochs 30 --batch-size 24 --amp
```

如模型已经训练完成，可只重新评估：

```bash
python src/train.py --experiment fusion --eval-only --batch-size 24 --amp
```

当前测试集结果：

| 实验组 | Accuracy | Macro F1 | Weighted F1 | Macro AUC OvR |
|---|---:|---:|---:|---:|
| `meta_only` | 0.3494 | 0.1818 | 0.4313 | 0.7409 |
| `image_only` | 0.8146 | 0.6932 | 0.8205 | 0.9621 |
| `fusion` | 0.8226 | 0.6914 | 0.8286 | 0.9652 |

模型接入与推理调用细节见本 README「运行应用」与「模型权重下载」小节，以及 `src/inference/` 推理接口实现。

## 数据接口

`src/data/dataset.py` 提供：

- `HAM10000Dataset`
- `MetadataEncoder`
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
| `lesion_id` | 病灶级分组字段，避免数据泄漏 |

## 团队分工

| 成员 | 角色 | 主要职责 |
|---|---|---|
| 组员 1 | 数据工程负责人 | 数据获取与检查、元数据清洗、图像预处理、数据集划分、EDA、类别不平衡处理 |
| 组员 2 | 算法与评估负责人 | 模型实现与训练、元数据融合、消融实验、指标计算、结果分析 |
| 组员 3 | 工程与交付负责人 | Streamlit 工程应用、推理接口整合、项目结构维护、报告、PPT、演示视频 |

## 项目文档

- `docs/数据分析与数据挖掘期末项目计划书.md`：课程项目计划书
- `docs/feature_engineering_liujiye.md`：数据预处理与特征工程说明
- `docs/团队分工声明.md`：团队成员分工声明
- `docs/dataset_check_summary.txt`：数据集检查汇总
- `reports/项目报告.md` / `reports/项目报告.pdf`：项目最终报告

## 注意事项

- `data/raw/` 和压缩包体积较大，不应提交到 Git。
- `outputs/` 和 `*.pth` 模型权重不提交到 Git，应作为单独附件或 GitHub Release 资产交付。
- HAM10000 类别分布不均衡，训练和评估时应关注 Macro F1、类别召回率和混淆矩阵。
- 任何真实医学判断均应由专业医生完成。
