"""Diagnosis workflow for Streamlit."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import PercentFormatter

from app.ui import configure_page, render_page_header, render_sidebar
from src.inference import PatientMetadata, predict
from src.inference.schema import CLASS_LABELS, LOCALIZATION_OPTIONS, MODEL_VARIANTS, SEX_OPTIONS


CLASS_NAMES = {
    "akiec": "光化性角化病",
    "bcc": "基底细胞癌",
    "bkl": "良性角化病变",
    "df": "皮肤纤维瘤",
    "mel": "黑色素瘤",
    "nv": "黑色素细胞痣",
    "vasc": "血管性病变",
}
MODEL_NAMES = {
    "image_only": "纯图像模型",
    "meta_only": "纯元数据模型",
    "fusion": "图像与元数据融合模型",
}
SEX_NAMES = {"female": "女性", "male": "男性", "unknown": "未知"}


def render_probability_chart(probabilities: dict[str, float], predicted_class: str) -> None:
    """Render a compact horizontal probability chart."""
    ordered_labels = sorted(CLASS_LABELS, key=probabilities.get)
    values = [probabilities[label] for label in ordered_labels]
    colors = ["#0f766e" if label == predicted_class else "#cbd5e1" for label in ordered_labels]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(ordered_labels, values, color=colors)
    ax.set_xlim(0, max(0.3, max(values) * 1.28))
    ax.set_xlabel("Model probability")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for index, value in enumerate(values):
        ax.text(value + 0.01, index, f"{value:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


configure_page("诊断")
render_sidebar()

render_page_header(
    "Model Integration Workspace",
    "皮肤病变诊断",
    "上传皮肤镜图像并补充患者元数据。真实 PyTorch 推理适配器接入后，本页将展示模型输出的七分类概率。",
)
st.markdown(
    '<div class="skinsight-warning"><strong>真实模型待接入。</strong>'
    "接入 PyTorch 适配器前不会生成虚拟医学结果。</div>",
    unsafe_allow_html=True,
)

form_column, result_column = st.columns([0.95, 1.05], gap="large")

with form_column:
    with st.container(border=True):
        st.markdown("### 1. 上传病变图像")
        uploaded_file = st.file_uploader("上传皮肤镜图片", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="已上传图片", width="stretch")

        st.markdown("### 2. 患者信息（元数据）")
        age = st.number_input("年龄", min_value=0, max_value=120, value=50, step=1)
        sex = st.selectbox("性别", options=SEX_OPTIONS, format_func=SEX_NAMES.get)
        localization = st.selectbox("病灶部位", options=LOCALIZATION_OPTIONS, index=2)

        st.markdown("### 3. 模型类型（选择推理分支）")
        variant = st.radio(
            "模型类型",
            options=MODEL_VARIANTS,
            format_func=MODEL_NAMES.get,
            index=2,
            label_visibility="collapsed",
        )
        submit = st.button("开始预测", type="primary", width="stretch")

        if submit:
            try:
                metadata = PatientMetadata(age=float(age), sex=sex, localization=localization)
                image = uploaded_file.getvalue() if uploaded_file is not None else None
                st.session_state["diagnosis_result"] = predict(
                    image=image,
                    metadata=metadata,
                    variant=variant,
                )
            except ValueError as exc:
                st.session_state.pop("diagnosis_result", None)
                st.error(str(exc))

with result_column:
    with st.container(border=True):
        st.markdown("### 预测结果")
        result = st.session_state.get("diagnosis_result")
        if result is None:
            st.info("填写左侧信息并生成结果后，此处将展示七分类概率。")
            st.markdown(
                """
                <div class="skinsight-placeholder">
                    <div class="skinsight-placeholder-title">等待推理输入</div>
                    <div class="skinsight-placeholder-copy">
                        真实模型适配器接入后，图像分支和融合分支需要上传皮肤镜图片。
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if result.warning:
                st.warning(result.warning)
            predicted_name = CLASS_NAMES[result.predicted_class]
            st.metric("最高概率类别", f"{result.predicted_class} - {predicted_name}")
            render_probability_chart(result.probabilities, result.predicted_class)
            st.caption(f"推理提供者：{result.provider}　模型类型：{result.variant}")

st.divider()
st.caption("仅用于课程演示，不能替代医生诊断。任何真实医学判断均应由专业医生完成。")
