"""Placeholder for model evaluation artifacts."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui import configure_page, render_page_header, render_placeholder, render_sidebar


configure_page("模型评估")
render_sidebar()

render_page_header(
    "Model Evaluation",
    "模型评估",
    "集中呈现三组模型在测试集上的量化表现，并展示融合模型的分类误差结构。",
)
st.markdown(
    '<div class="skinsight-note">页面骨架已建立。真实模型交付后，将从评估产物中读取指标与图表。</div>',
    unsafe_allow_html=True,
)

columns = st.columns(3, gap="medium")
for column, label in zip(columns, ["image_only", "meta_only", "fusion"]):
    with column:
        st.metric(label, "等待评估")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("### Macro F1 对比")
    render_placeholder("等待指标文件", "将展示三组实验的 Macro F1 对比表格与摘要。")
with right:
    st.markdown("### 混淆矩阵")
    render_placeholder("等待融合模型评估图", "将展示 fusion 模型在测试集上的混淆矩阵热力图。")

st.markdown("### One-vs-Rest ROC 曲线")
render_placeholder("等待 ROC 图表", "将按七个类别展示 ROC 曲线和对应 AUC。")
