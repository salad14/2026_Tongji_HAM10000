"""Shared Streamlit layout and visual helpers."""

import streamlit as st


def configure_page(title: str) -> None:
    """Configure one Streamlit page and apply shared styles."""
    st.set_page_config(page_title=f"{title} | SkinSight", page_icon="SS", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --skinsight-accent: #087f7b;
            --skinsight-accent-dark: #05635f;
            --skinsight-accent-soft: #edf9f8;
            --skinsight-text: #17223b;
            --skinsight-muted: #64748b;
            --skinsight-border: #dce7ea;
            --skinsight-panel: #ffffff;
            --skinsight-warning: #a75b00;
        }
        [data-testid="stAppViewContainer"] {
            background: #ffffff;
            color: var(--skinsight-text);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background: #f6f9fa;
            border-right: 1px solid var(--skinsight-border);
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 1.35rem;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--skinsight-text);
            letter-spacing: -0.02em;
        }
        h1 {
            font-size: 2.45rem !important;
            margin-bottom: 0.45rem !important;
        }
        h3 {
            margin-top: 0.8rem !important;
        }
        p, label, [data-testid="stCaptionContainer"] {
            line-height: 1.65;
        }
        .skinsight-brand {
            color: var(--skinsight-accent);
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.2rem;
        }
        .skinsight-subtitle {
            color: var(--skinsight-muted);
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .skinsight-kicker {
            color: var(--skinsight-accent-dark);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        .skinsight-lead {
            color: #526176;
            font-size: 1rem;
            line-height: 1.8;
            margin-bottom: 1rem;
            max-width: 880px;
        }
        .skinsight-note {
            border-left: 4px solid var(--skinsight-accent);
            background: var(--skinsight-accent-soft);
            color: #115e59;
            padding: 0.95rem 1.05rem;
            margin: 0.85rem 0 1.35rem;
            border-radius: 0 0.45rem 0.45rem 0;
        }
        .skinsight-warning {
            border-left: 4px solid #f59e0b;
            background: #fffbeb;
            color: var(--skinsight-warning);
            padding: 0.95rem 1.05rem;
            margin: 0.85rem 0 1.35rem;
            border-radius: 0 0.45rem 0.45rem 0;
        }
        .skinsight-hero {
            padding: 1.25rem 1.35rem;
            border: 1px solid var(--skinsight-border);
            border-radius: 0.85rem;
            background: linear-gradient(135deg, #ffffff 0%, #f5fbfb 100%);
            box-shadow: 0 12px 30px rgba(15, 67, 76, 0.055);
            margin: 0.7rem 0 1.35rem;
        }
        .skinsight-card {
            min-height: 156px;
            padding: 1.15rem 1.2rem;
            border: 1px solid var(--skinsight-border);
            border-radius: 0.8rem;
            background: var(--skinsight-panel);
            box-shadow: 0 8px 24px rgba(15, 67, 76, 0.045);
        }
        .skinsight-card-number {
            color: var(--skinsight-accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            margin-bottom: 0.55rem;
        }
        .skinsight-card-title {
            color: var(--skinsight-text);
            font-size: 1.08rem;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }
        .skinsight-card-copy {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.7;
        }
        .skinsight-placeholder {
            margin-top: 1rem;
            padding: 2rem 1.5rem;
            border: 1px dashed #b8ced2;
            border-radius: 0.85rem;
            background: #fbfefe;
            text-align: center;
        }
        .skinsight-placeholder-title {
            color: var(--skinsight-accent-dark);
            font-weight: 750;
            margin-bottom: 0.4rem;
        }
        .skinsight-placeholder-copy {
            color: var(--skinsight-muted);
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--skinsight-border);
            border-radius: 0.8rem;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 7px 22px rgba(15, 67, 76, 0.04);
        }
        div[data-testid="stMetricValue"] {
            color: var(--skinsight-text);
        }
        [data-testid="stFileUploaderDropzone"],
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            background: #f8fafc;
            border-color: var(--skinsight-border);
        }
        [data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, var(--skinsight-accent), var(--skinsight-accent-dark));
            border: 0;
            box-shadow: 0 8px 18px rgba(8, 127, 123, 0.18);
            font-weight: 700;
        }
        [data-testid="stButton"] button[kind="primary"]:hover {
            background: var(--skinsight-accent-dark);
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            border-radius: 0.55rem;
            color: #526176;
            font-weight: 650;
            margin-bottom: 0.15rem;
            padding: 0.56rem 0.7rem;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
            background: #e8f4f4;
            color: var(--skinsight-accent-dark);
        }
        [data-testid="stSidebar"] hr {
            margin: 1.15rem 0;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1.4rem;
            }
            h1 {
                font-size: 2rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_page_link(page: str, label: str, icon: str) -> None:
    """Render navigation, falling back when a page is tested standalone."""
    try:
        st.sidebar.page_link(page, label=label, icon=icon)
    except KeyError:
        st.sidebar.markdown(f"- {label}")


def render_sidebar() -> None:
    """Render persistent project context in the sidebar."""
    st.sidebar.markdown('<div class="skinsight-brand">SkinSight</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="skinsight-subtitle">皮肤病变智能诊断系统<br>数据分析与数据挖掘课程项目</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown('<div class="skinsight-kicker">Research Console</div>', unsafe_allow_html=True)
    _sidebar_page_link("main.py", "首页", ":material/home:")
    _sidebar_page_link("pages/1_diagnosis.py", "病变诊断", ":material/stethoscope:")
    _sidebar_page_link("pages/2_model_evaluation.py", "模型评估", ":material/query_stats:")
    _sidebar_page_link("pages/3_ablation_study.py", "消融实验", ":material/compare_arrows:")
    _sidebar_page_link("pages/4_eda.py", "数据探索", ":material/analytics:")
    st.sidebar.divider()
    st.sidebar.caption("真实模型已接入；CUDA 可用时默认使用 GPU 推理。")
    st.sidebar.caption("仅用于课程学习，不能替代医生诊断。")


def render_page_header(kicker: str, title: str, description: str) -> None:
    """Render a consistent academic dashboard heading."""
    st.markdown(f'<div class="skinsight-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="skinsight-lead">{description}</div>', unsafe_allow_html=True)


def render_placeholder(title: str, description: str) -> None:
    """Render an explicit placeholder for model-dependent sections."""
    st.markdown(
        f"""
        <div class="skinsight-placeholder">
            <div class="skinsight-placeholder-title">{title}</div>
            <div class="skinsight-placeholder-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
