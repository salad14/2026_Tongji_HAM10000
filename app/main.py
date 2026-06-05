"""SkinSight Streamlit application entrypoint."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui import configure_page, render_page_header, render_sidebar


configure_page("首页")
render_sidebar()

render_page_header(
    "Multimodal Dermatology Research",
    "SkinSight",
    "面向 HAM10000 的多模态皮肤病变分类与数据分析平台。"
    "系统计划融合皮肤镜图像与患者元数据，以可解释的实验流程呈现七分类研究结果。",
)
st.markdown(
    '<div class="skinsight-warning"><strong>当前为工程联调版本。</strong>'
    "真实模型推理适配器尚未接入，诊断页不会生成虚拟预测结果。</div>",
    unsafe_allow_html=True,
)

columns = st.columns(4, gap="medium")
cards = [
    ("01", "病变诊断", "上传图像并填写元数据，提前验证统一推理接口与七分类结果展示。"),
    ("02", "模型评估", "预留 Macro F1、混淆矩阵和 ROC 曲线展示区域，等待模型交付。"),
    ("03", "消融实验", "预留三分支同图对比流程，用于展示多模态融合效果。"),
    ("04", "数据探索", "直接复用现有 CSV 与 EDA 图表，查看样本规模和类别不平衡。"),
]
for column, (number, title, copy) in zip(columns, cards):
    with column:
        st.markdown(
            f"""
            <div class="skinsight-card">
                <div class="skinsight-card-number">{number}</div>
                <div class="skinsight-card-title">{title}</div>
                <div class="skinsight-card-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### 当前可用范围")
st.markdown(
    '<div class="skinsight-note">诊断页已保留上传、元数据表单与统一推理接口，数据探索页已接入现有数据材料。'
    "模型评估与消融实验页面已建立骨架，将在真实模型适配器接入后填充结果。</div>",
    unsafe_allow_html=True,
)
st.caption("本项目仅用于课程学习与辅助研究展示，不能替代医生诊断。")
