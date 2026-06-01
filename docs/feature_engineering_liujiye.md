# 数据预处理与特征工程说明（刘继业 2252752）

## 1. 数据工程任务说明

刘继业负责本项目的数据工程部分，主要工作包括 HAM10000 数据集下载、数据完整性检查、元数据清洗、病灶级数据集划分、EDA 图表生成、图像预处理接口封装，以及类别不平衡处理工具实现。该部分输出的数据文件和代码接口可直接交付给算法同学用于后续模型训练。

## 2. HAM10000 数据字段说明

- `lesion_id`：病灶编号。同一病灶可能对应多张图片，划分数据集时必须以该字段做分组。
- `image_id`：图片编号，可用于匹配具体的 jpg 图像文件。
- `dx`：疾病诊断类别缩写，包括 `akiec`、`bcc`、`bkl`、`df`、`mel`、`nv`、`vasc`。
- `dx_type`：诊断确认方式，例如 histo、follow_up、consensus、confocal。
- `age`：患者年龄。
- `sex`：患者性别。
- `localization`：病灶所在身体部位。
- `image_path`：清洗阶段新增字段，表示图片相对项目根目录的路径。
- `label`：清洗阶段新增字段，将 `dx` 映射为整数标签。
- `dx_full_name`：清洗阶段新增字段，表示疾病类别英文全称。

标签映射如下：

| dx | label | dx_full_name |
| --- | ---: | --- |
| akiec | 0 | Actinic keratoses and intraepithelial carcinoma |
| bcc | 1 | Basal cell carcinoma |
| bkl | 2 | Benign keratosis-like lesions |
| df | 3 | Dermatofibroma |
| mel | 4 | Melanoma |
| nv | 5 | Melanocytic nevi |
| vasc | 6 | Vascular lesions |

## 3. 数据清洗策略

首先根据 `image_id` 在 `HAM10000_images_part_1` 和 `HAM10000_images_part_2` 中匹配对应 jpg 文件，并新增 `image_path` 字段。无法匹配图片路径的记录会被删除，避免后续训练时出现图片读取失败。

其次删除重复的 `image_id`，保证每条样本记录对应唯一图像。`age` 的缺失值使用清洗后数据中的年龄中位数填补，降低异常缺失对训练流程的影响。`sex` 和 `localization` 中的缺失值或 unknown 类别统一保留为 `unknown`，因为这些信息本身可以表达元数据不确定性，不宜简单删除样本。

最后将 `dx` 映射为整数 `label`，并新增 `dx_full_name`，便于模型训练和报告展示同时使用。

## 4. 数据集划分策略

HAM10000 中同一 `lesion_id` 可能包含多张图片。如果使用普通随机划分，同一病灶的不同图片可能同时进入训练集和测试集，导致病灶级数据泄漏，使测试结果虚高。因此本项目不能使用普通随机划分。

本项目使用 `sklearn.model_selection.GroupShuffleSplit`，以 `lesion_id` 作为 group，将数据划分为 train、val、test，比例约为 70%、15%、15%。划分后会检查 train、val、test 之间的 `lesion_id` 是否存在交集；如果发现交集，脚本会直接报错终止。

## 5. 图像预处理策略

图像读取阶段统一使用 RGB 格式，并将图片 resize 到 224x224，以适配常见 CNN 和 ViT 主干网络。归一化采用 ImageNet mean/std：

- mean = `(0.485, 0.456, 0.406)`
- std = `(0.229, 0.224, 0.225)`

训练集可以使用随机增强，包括水平翻转、随机旋转、平移缩放旋转、亮度和对比度扰动等，以提高模型泛化能力。验证集和测试集不能使用随机数据增强，只进行 resize、归一化和 tensor 转换，保证评估结果稳定可复现。

## 6. 类别不平衡处理

HAM10000 存在明显类别不平衡，其中 `nv` 类样本数量最多，部分类别样本较少。训练阶段建议使用以下方式处理：

- 类别权重：根据训练集标签分布计算 class weights，并传入损失函数。
- `WeightedRandomSampler`：为少数类样本分配更高采样权重，使训练 batch 中少数类更容易被采样。

不建议直接对原始图像使用 SMOTE。SMOTE 更适合低维结构化特征，不适合直接对原始高维图像像素做插值；对图像像素插值可能生成不符合真实皮肤病变形态的样本。本项目图像训练阶段主要采用类别权重和 `WeightedRandomSampler`，纯元数据模型可以额外尝试 SMOTE 作为对比。

## 7. 交付给算法同学的接口说明

清洗和划分完成后，算法同学主要使用以下文件：

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/split.csv`

CSV 中关键字段如下：

- `image_path`：图片路径，读取图像时使用。
- `label`：整数类别标签，训练分类模型时使用。
- `age`、`sex`、`localization`：可作为多模态融合模型中的元数据特征。

代码接口位于 `src/data/dataset.py`：

- `HAM10000Dataset`：PyTorch Dataset，支持 `use_metadata=True/False`。
- `get_train_transform()`：训练集图像增强和归一化。
- `get_valid_transform()`：验证集和测试集确定性预处理。
- `compute_class_weights(csv_path)`：计算类别权重。
- `build_weighted_sampler(csv_path)`：构造类别不平衡采样器。

## 8. 可放入项目报告的数据预处理章节文字

本项目使用 HAM10000 皮肤病变数据集作为实验数据。数据工程阶段首先通过 Kaggle API 下载原始图像和元数据，并对数据完整性进行检查，包括图片数量、metadata 字段、缺失图片、重复 `image_id`、重复 `lesion_id`、类别分布以及关键字段缺失情况。随后根据 `image_id` 匹配图像路径，删除无法匹配图片的记录和重复图片记录，对年龄缺失值使用中位数填补，并将性别和病灶部位中的缺失或 unknown 统一保留为 `unknown` 类别。

在标签处理方面，将原始诊断字段 `dx` 映射为 0 到 6 的整数标签，并保留疾病英文全称用于结果解释。考虑到 HAM10000 中同一病灶可能对应多张图片，数据集划分采用基于 `lesion_id` 的分组划分策略，而不是普通随机划分，从而避免同一病灶出现在训练集和测试集中造成数据泄漏。最终按照约 70%、15%、15% 的比例生成训练集、验证集和测试集。

图像预处理阶段将所有图像转换为 RGB 格式并 resize 到 224x224，随后使用 ImageNet mean/std 进行归一化。训练集使用随机翻转、随机旋转、平移缩放旋转、亮度和对比度扰动等增强方法，以提高模型泛化能力；验证集和测试集只采用确定性预处理，不使用随机增强。针对类别不平衡问题，训练阶段提供类别权重和 `WeightedRandomSampler` 两种处理方式。由于 SMOTE 主要适用于低维结构化特征，不适合直接对高维原始图像像素进行插值，因此图像模型训练中不直接使用 SMOTE。
