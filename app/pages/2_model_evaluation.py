"""Model evaluation dashboard backed by training outputs."""

import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.ui import configure_page, render_page_header, render_sidebar


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS = ("meta_only", "image_only", "fusion")


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        path = OUTPUTS_DIR / experiment / "test_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "实验组": experiment,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Macro AUC OvR": metrics["macro_auc_ovr"],
                "Test Loss": metrics["test_loss"],
            }
        )
    return pd.DataFrame(rows)


configure_page("模型评估")
render_sidebar()

render_page_header(
    "Model Evaluation",
    "模型评估",
    "集中呈现三组模型在测试集上的量化表现，并展示融合模型的分类误差结构。",
)
st.markdown(
    '<div class="skinsight-note">本页读取 <code>outputs/*/test_metrics.json</code> '
    "和混淆矩阵图，展示三组消融实验的真实测试集结果。</div>",
    unsafe_allow_html=True,
)

metrics_df = load_metrics()
if metrics_df.empty:
    st.error("未找到模型评估指标，请确认 outputs/*/test_metrics.json 已放在项目根目录。")
    st.stop()

macro_f1 = metrics_df.set_index("实验组")["Macro F1"]
accuracy = metrics_df.set_index("实验组")["Accuracy"]
columns = st.columns(3, gap="medium")
for column, experiment in zip(columns, EXPERIMENTS):
    with column:
        st.metric(
            experiment,
            f"{macro_f1.get(experiment, 0):.4f}",
            help="测试集 Macro F1",
        )

st.markdown("### 测试集指标")
display_df = metrics_df.copy()
for column in ["Accuracy", "Macro F1", "Weighted F1", "Macro AUC OvR", "Test Loss"]:
    display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
st.dataframe(display_df, width="stretch", hide_index=True)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("### 关键结论")
    fusion_accuracy = accuracy.get("fusion", 0)
    image_accuracy = accuracy.get("image_only", 0)
    st.markdown(
        f"""
        <div class="skinsight-card">
            <div class="skinsight-card-title">推荐接入 fusion 模型</div>
            <div class="skinsight-card-copy">
            fusion 的 Accuracy 为 {fusion_accuracy:.4f}，image_only 为 {image_accuracy:.4f}。
            融合模型在 Accuracy、Weighted F1 和 Macro AUC 上略优于纯图像模型；
            Macro F1 基本持平，说明元数据提供了辅助信息，但少数类平均提升有限。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown("### fusion 混淆矩阵")
    fusion_matrix = OUTPUTS_DIR / "fusion" / "confusion_matrix.png"
    if fusion_matrix.exists():
        st.image(str(fusion_matrix), caption="fusion 测试集混淆矩阵", width="stretch")
    else:
        st.info("暂未找到 outputs/fusion/confusion_matrix.png")

st.markdown("### 三组混淆矩阵")
tabs = st.tabs(list(EXPERIMENTS))
for tab, experiment in zip(tabs, EXPERIMENTS):
    with tab:
        matrix_path = OUTPUTS_DIR / experiment / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption=f"{experiment} confusion matrix", width="stretch")
        else:
            st.info(f"暂未找到 outputs/{experiment}/confusion_matrix.png")
