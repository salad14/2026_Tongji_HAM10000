"""Ablation comparison page using trained models."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import PercentFormatter

from app.ui import configure_page, render_page_header, render_sidebar
from src.inference import PatientMetadata, predict, select_device
from src.inference.schema import CLASS_LABELS, LOCALIZATION_OPTIONS, MODEL_VARIANTS, SEX_OPTIONS


CLASS_NAMES = {
    "akiec": "光化性角化病",
    "bcc": "基底细胞癌",
    "bkl": "良性角化病变",
    "df": "皮肤纤维瘤",
    "mel": "黑色素瘤",
    "nv": "色素痣",
    "vasc": "血管病变",
}
SEX_NAMES = {"female": "女性", "male": "男性", "unknown": "未知"}


def render_probability_chart(probabilities: dict[str, float], predicted_class: str) -> None:
    ordered_labels = sorted(CLASS_LABELS, key=probabilities.get)
    values = [probabilities[label] for label in ordered_labels]
    colors = ["#0f766e" if label == predicted_class else "#cbd5e1" for label in ordered_labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(ordered_labels, values, color=colors)
    ax.set_xlim(0, max(0.3, max(values) * 1.28))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


configure_page("消融实验")
render_sidebar()

render_page_header(
    "Ablation Study",
    "消融实验",
    "对同一病例运行图像分支、元数据分支和融合分支，直观比较不同输入模态对预测结果的影响。",
)
st.markdown(
    f'<div class="skinsight-note"><strong>真实三分支模型已接入。</strong>'
    f"当前默认推理设备：<code>{select_device()}</code>。"
    "上传同一张图片后，将并排展示 image_only、meta_only 和 fusion 的输出。</div>",
    unsafe_allow_html=True,
)

input_column, preview_column = st.columns([0.9, 1.1], gap="large")
with input_column:
    with st.container(border=True):
        st.markdown("### 输入病例")
        uploaded_file = st.file_uploader("上传皮肤镜图片", type=["jpg", "jpeg", "png"])
        age = st.number_input("年龄", min_value=0, max_value=120, value=50, step=1)
        sex = st.selectbox("性别", options=SEX_OPTIONS, format_func=SEX_NAMES.get)
        localization = st.selectbox("病灶部位", options=LOCALIZATION_OPTIONS, index=2)
        submit = st.button("运行三分支对比", type="primary", width="stretch")
with preview_column:
    if uploaded_file is not None:
        st.image(uploaded_file, caption="用于三分支对比的同一张输入图片", width="stretch")
    else:
        st.info("请先上传图片。image_only 与 fusion 分支需要图像输入。")

if submit:
    if uploaded_file is None:
        st.error("三分支对比需要上传同一张皮肤镜图片。")
    else:
        metadata = PatientMetadata(age=float(age), sex=sex, localization=localization)
        image = uploaded_file.getvalue()
        try:
            with st.spinner("正在运行三组模型..."):
                st.session_state["ablation_results"] = {
                    variant: predict(image=image, metadata=metadata, variant=variant)
                    for variant in MODEL_VARIANTS
                }
        except Exception as exc:
            st.session_state.pop("ablation_results", None)
            st.error(f"三分支对比运行失败：{exc}")

results = st.session_state.get("ablation_results")
if results:
    st.markdown("### 同图三分支对比")
    columns = st.columns(3, gap="medium")
    for column, variant in zip(columns, MODEL_VARIANTS):
        result = results[variant]
        predicted_name = CLASS_NAMES[result.predicted_class]
        top_probability = result.probabilities[result.predicted_class]
        with column:
            with st.container(border=True):
                st.markdown(f"#### {variant}")
                st.metric("最高概率类别", f"{result.predicted_class} - {predicted_name}")
                st.metric("最高概率", f"{top_probability:.1%}")
                render_probability_chart(result.probabilities, result.predicted_class)
                st.caption(f"推理提供者：{result.provider}")
