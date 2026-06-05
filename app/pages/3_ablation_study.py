"""Placeholder for multimodal ablation comparison."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui import configure_page, render_page_header, render_placeholder, render_sidebar


configure_page("消融实验")
render_sidebar()

render_page_header(
    "Ablation Study",
    "消融实验",
    "对同一病例运行图像分支、元数据分支和融合分支，直观比较不同输入模态对预测结果的影响。",
)
st.markdown(
    '<div class="skinsight-warning"><strong>等待真实模型接入。</strong>'
    "本页暂不生成虚拟对比，避免将占位数值误认为消融实验结论。</div>",
    unsafe_allow_html=True,
)

st.markdown("### 同图三分支对比")
columns = st.columns(3, gap="medium")
for column, (number, title, copy) in zip(
    columns,
    [
        ("01", "图像分支", "image_only：仅使用皮肤镜图像。"),
        ("02", "元数据分支", "meta_only：仅使用年龄、性别与部位。"),
        ("03", "融合分支", "fusion：联合图像特征与结构化元数据。"),
    ],
):
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

render_placeholder("等待三个模型检查点", "模型交付后，此处将加入图片上传、统一元数据表单与三组概率并排比较。")
