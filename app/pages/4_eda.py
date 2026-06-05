"""EDA dashboard backed by committed HAM10000 artifacts."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.ui import configure_page, render_page_header, render_sidebar


SPLIT_CSV = PROJECT_ROOT / "data" / "processed" / "split.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


@st.cache_data(show_spinner=False)
def load_split_data(csv_path: str) -> pd.DataFrame:
    """Load committed split metadata once per application session."""
    return pd.read_csv(csv_path)


def render_figure(filename: str, caption: str) -> None:
    """Render an EDA figure or show a local missing-file message."""
    figure_path = FIGURE_DIR / filename
    if figure_path.exists():
        st.image(str(figure_path), caption=caption, width="stretch")
    else:
        st.info(f"暂未找到图表：reports/figures/{filename}")


configure_page("数据探索")
render_sidebar()

render_page_header(
    "Exploratory Data Analysis",
    "数据探索",
    "基于已清洗 HAM10000 元数据与现有分析图表，快速浏览样本规模、类别分布和关键字段特征。",
)
st.markdown(
    '<div class="skinsight-note">本页直接读取仓库中的处理后 CSV 和 EDA 图表，'
    "不依赖原始图片目录。</div>",
    unsafe_allow_html=True,
)

if not SPLIT_CSV.exists():
    st.error("未找到 data/processed/split.csv，请先完成数据处理。")
    st.stop()

df = load_split_data(str(SPLIT_CSV))
split_counts = df["split"].value_counts()

metric_columns = st.columns(4)
metric_columns[0].metric("样本总数", f"{len(df):,}")
metric_columns[1].metric("训练集", f"{split_counts.get('train', 0):,}")
metric_columns[2].metric("验证集", f"{split_counts.get('val', 0):,}")
metric_columns[3].metric("测试集", f"{split_counts.get('test', 0):,}")

st.markdown("### 类别分布")
st.warning("HAM10000 存在明显类别不平衡。训练与评估应重点关注 Macro F1，不能只看 Accuracy。")
class_summary = (
    df["dx"]
    .value_counts()
    .rename_axis("类别")
    .reset_index(name="样本数")
)
class_summary["占比"] = class_summary["样本数"].div(len(df)).map(lambda value: f"{value:.1%}")
st.dataframe(class_summary, width="stretch", hide_index=True)
render_figure("class_distribution.png", "HAM10000 类别分布")

st.markdown("### 样例图像")
render_figure("sample_images_by_class.png", "每个诊断类别的一张样例图像")

st.markdown("### 元数据分布")
left, right = st.columns(2, gap="large")
with left:
    render_figure("age_distribution.png", "年龄分布")
    render_figure("localization_distribution.png", "病灶部位分布")
with right:
    render_figure("sex_distribution.png", "性别分布")
    render_figure("dx_type_distribution.png", "诊断确认方式分布")

st.markdown("### 缺失值")
render_figure("missing_values.png", "清洗后字段缺失值统计")
