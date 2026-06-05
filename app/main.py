"""SkinSight Streamlit application entrypoint."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui import configure_page, render_page_header, render_sidebar
from src.inference import select_device


configure_page("首页")
render_sidebar()

render_page_header(
    "Multimodal Dermatology Research",
    "SkinSight",
    "面向 HAM10000 的多模态皮肤病变分类与数据分析平台。系统融合皮肤镜图像与患者元数据，"
    "展示七分类模型推理、评估结果、消融对比和数据探索流程。",
)

st.markdown(
    f'<div class="skinsight-note"><strong>真实模型已接入。</strong>'
    f"当前默认推理设备：<code>{select_device()}</code>。"
    "诊断页、模型评估页和消融实验页可调用</div>",
    unsafe_allow_html=True,
)

columns = st.columns(4, gap="medium")
cards = [
    ("01", "病变诊断", "上传图像并填写元数据，调用统一推理接口输出 HAM10000 七分类概率。"),
    ("02", "模型评估", "读取各分支 test_metrics.json 与混淆矩阵，展示 Macro F1、Accuracy 等测试集结果。"),
    ("03", "消融实验", "对同一病例并行运行 image_only、meta_only 和 fusion，比较不同输入模态的预测差异。"),
    ("04", "数据探索", "复用现有 CSV 与 EDA 图表，查看样本规模、类别分布和元数据统计。"),
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


st.caption("本项目仅用于课程学习与辅助研究展示，不能替代医生诊断。")
