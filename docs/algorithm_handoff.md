# 算法模型交付与工程接入说明

本文档面向 Streamlit 应用开发、代码整合、报告/PPT/演示视频制作环节，说明算法阶段已完成的内容、模型文件交付方式、工程接入流程和实验结果使用方式。

## 1. 算法阶段完成内容

算法阶段已完成以下工作：

- 实现三组消融实验模型：
  - `meta_only`：仅使用年龄、性别、病灶部位。
  - `image_only`：仅使用皮肤镜图像，主干为 EfficientNet-B0。
  - `fusion`：同时使用图像和元数据，为推荐接入系统的最终模型。
- 实现训练与评估脚本：`src/train.py`。
- 实现单张图片推理脚本：`src/inference.py`。
- 修复元数据编码方式，使 train/val/test 使用同一套编码规则。
- 完成三组模型训练，并重新评估测试集指标。
- 生成模型权重、训练过程 CSV、测试指标 JSON 和混淆矩阵图片。

Streamlit 系统推荐默认使用：

```text
outputs/fusion/best_model.pth
```

该模型输入为：

```text
皮肤镜图像 + 年龄 + 性别 + 病灶部位
```

输出为 7 类病变概率。

## 2. 需要额外交付的模型文件

GitHub 仓库包含代码和文档，但模型权重不会直接提交到 Git，因为 `.pth` 文件较大。

工程接入至少需要以下目录：

```text
outputs/fusion/
├── best_model.pth
├── config.json
├── confusion_matrix.png
├── history.csv
└── test_metrics.json
```

如需实现完整消融实验展示页，还需要以下三个目录：

```text
outputs/meta_only/
outputs/image_only/
outputs/fusion/
```

上述目录应放在项目根目录下：

```text
2026_Tongji_HAM10000/
└── outputs/
    ├── meta_only/
    ├── image_only/
    └── fusion/
```

## 3. 环境准备

创建并激活项目环境：

```powershell
conda create -n skinsight python=3.10 -y
conda activate skinsight
pip install -r requirements.txt
```

如本地使用 NVIDIA GPU，可运行以下命令确认 PyTorch 能识别 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

如果只做 Streamlit 推理演示，CPU 也可运行，但模型加载和预测速度会较慢。

## 4. 数据准备

如果仅实现“上传单张图片并预测”的 Streamlit 页面，不需要完整原始数据集，只需要用户上传的图片和模型权重。

如果需要展示测试集评估、随机样本、EDA 页面，或重新评估模型，则需要原始图片目录：

```text
data/raw/ham10000/
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

如果本地没有原始图片，先配置 Kaggle access token，然后运行：

```powershell
python src/data/download_kaggle.py
```

仓库中已有处理后的划分文件：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

这些 CSV 用于训练、评估和测试集展示。

## 5. 模型加载检查

接入 Streamlit 前，建议先运行一次命令行推理，确认模型、权重、图像预处理和元数据编码均可正常工作：

```powershell
python src/inference.py `
  --checkpoint outputs/fusion/best_model.pth `
  --experiment fusion `
  --image data/raw/ham10000/HAM10000_images_part_1/ISIC_0027419.jpg `
  --age 80 `
  --sex male `
  --localization scalp
```

成功时会得到类似 JSON 输出：

```json
{
  "prediction": "bkl",
  "probabilities": {
    "akiec": 0.00001,
    "bcc": 0.00001,
    "bkl": 0.99937,
    "df": 0.00000,
    "mel": 0.00031,
    "nv": 0.00031,
    "vasc": 0.00000
  }
}
```

## 6. Streamlit 接入方式

`src/inference.py` 中提供两个可复用函数：

```python
from src.inference import load_model, predict_one
```

应用启动时加载一次模型：

```python
model, encoder, device = load_model(
    "outputs/fusion/best_model.pth",
    experiment="fusion",
)
```

用户上传图片并输入元数据后调用：

```python
probabilities = predict_one(
    model=model,
    encoder=encoder,
    device=device,
    image_path=image_path,
    age=age,
    sex=sex,
    localization=localization,
    experiment="fusion",
)
```

`probabilities` 返回字典：

```python
{
    "akiec": 0.01,
    "bcc": 0.02,
    "bkl": 0.80,
    "df": 0.01,
    "mel": 0.08,
    "nv": 0.07,
    "vasc": 0.01,
}
```

界面可展示：

- 预测类别：概率最大的类别。
- 7 类置信度条形图。
- 医疗免责声明：系统仅用于课程学习与辅助研究展示，不能替代医生诊断。

## 7. 前端输入项建议

诊断页面建议包含：

- 图片上传：皮肤镜图像。
- 年龄输入：数字输入或滑块，范围可设为 0 到 100。
- 性别选择：
  - `male`
  - `female`
  - `unknown`
- 病灶部位选择：
  - `abdomen`
  - `acral`
  - `back`
  - `chest`
  - `ear`
  - `face`
  - `foot`
  - `genital`
  - `hand`
  - `lower extremity`
  - `neck`
  - `scalp`
  - `trunk`
  - `unknown`
  - `upper extremity`

以上取值来自训练集元数据，前端传入模型时应保持拼写一致。

## 8. 模型输出类别顺序

模型输出固定为 7 类：

| index | dx | 英文全称 |
|---:|---|---|
| 0 | `akiec` | Actinic keratoses and intraepithelial carcinoma |
| 1 | `bcc` | Basal cell carcinoma |
| 2 | `bkl` | Benign keratosis-like lesions |
| 3 | `df` | Dermatofibroma |
| 4 | `mel` | Melanoma |
| 5 | `nv` | Melanocytic nevi |
| 6 | `vasc` | Vascular lesions |

界面、报告和 PPT 中应保持该顺序一致。

## 9. 可展示的实验结果

测试集指标如下：

| 实验 | Accuracy | Macro F1 | Weighted F1 | Macro AUC OvR |
|---|---:|---:|---:|---:|
| `meta_only` | 0.3494 | 0.1818 | 0.4313 | 0.7409 |
| `image_only` | 0.8146 | 0.6932 | 0.8205 | 0.9621 |
| `fusion` | 0.8226 | 0.6914 | 0.8286 | 0.9652 |

“模型评估”页面建议展示：

- 三组实验指标表。
- `outputs/fusion/confusion_matrix.png`。
- 纯图像模型与融合模型对比。

报告中可使用以下结论：

```text
纯元数据模型表现较弱，说明年龄、性别和病灶部位无法单独完成可靠诊断。纯图像 EfficientNet-B0 显著提升分类性能，是主要信息来源。融合模型在 Accuracy、Weighted F1 和 Macro AUC 上略优于纯图像模型，但 Macro F1 基本持平，说明元数据对整体预测有辅助作用，但对少数类平均性能提升有限。
```

## 10. 重新评估模型

无需重新训练，可直接加载已有 `best_model.pth` 重新计算测试集指标：

```powershell
python src/train.py --experiment meta_only --eval-only --batch-size 128
python src/train.py --experiment image_only --eval-only --batch-size 32 --amp
python src/train.py --experiment fusion --eval-only --batch-size 24 --amp
```

输出会覆盖：

```text
outputs/<experiment>/test_metrics.json
outputs/<experiment>/confusion_matrix.png
```

## 11. 重新训练模型

通常不建议重新训练，当前结果已可用于课程项目交付。如需重新训练，命令如下：

```powershell
python src/train.py --experiment meta_only --epochs 30 --batch-size 128
python src/train.py --experiment image_only --epochs 30 --batch-size 32 --amp
python src/train.py --experiment fusion --epochs 30 --batch-size 24 --amp
```

训练会更新：

```text
outputs/<experiment>/best_model.pth
outputs/<experiment>/history.csv
outputs/<experiment>/test_metrics.json
outputs/<experiment>/confusion_matrix.png
```

重新训练前应确认是否允许覆盖现有训练结果。

## 12. GitHub 提交注意事项

以下内容不要直接提交到 GitHub：

```text
data/raw/
outputs/
*.pth
*.pt
*.zip
```

原因：

- 原始图片和压缩包体积较大。
- 模型权重较大，建议通过网盘、课程平台附件或 GitHub Release 资产单独交付。
- `.gitignore` 已忽略这些内容。

## 13. 后续工程任务清单

建议后续工程交付按以下顺序推进：

1. 从 GitHub 拉取最新代码。
2. 获取 `outputs/fusion/best_model.pth`，并放回 `outputs/fusion/`。
3. 运行第 5 节的单图推理命令，确认模型可加载。
4. 创建 `app/` 目录并实现 Streamlit 页面。
5. 在诊断页面接入 `load_model()` 和 `predict_one()`。
6. 在评估页面展示三组指标表和混淆矩阵。
7. 在报告和 PPT 中使用第 9 节的指标和结论。
8. 最终演示时明确说明：系统仅用于课程学习与辅助研究展示，不能替代医生诊断。
