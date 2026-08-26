"""
Funções de layout e estruturação da página.
Gerencia seções, grids e organização visual.
"""

import streamlit as st
from ui.theme import COLORS

# =============================================================================
# SEÇÕES DA PÁGINA
# =============================================================================

def section_header(title: str, subtitle: str = "", icon: str = ""):
    """Cabeçalho de seção com título, subtítulo e ícone opcional."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <div class="section-title">
                {icon} {title.upper()}
                {f'<span style="float:right;font-weight:400;font-size:0.7rem;color:{COLORS["text_secondary"]};">{subtitle}</span>' if subtitle else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "", note: str = ""):
    """Card de métrica compacto."""
    delta_html = f'<span class="metric-delta">{delta}</span>' if delta else ""
    note_html = f'<div style="font-size:0.7rem;color:{COLORS["text_secondary"]};margin-top:0.2rem;">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="panel panel-accent">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(content, accent: bool = False):
    """Wrapper para painel com ou sem acento lateral."""
    cls = "panel panel-accent" if accent else "panel"
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)


def image_panel(img, caption: str = ""):
    """Exibe uma imagem dentro de um container estilizado."""
    caption_html = f'<div style="font-size:0.7rem;color:{COLORS["text_secondary"]};padding:0.3rem 0 0.2rem 0;text-align:center;">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="image-container">
            {img}
        </div>
        {caption_html}
        """,
        unsafe_allow_html=True,
    )


def pipeline_stage(label: str, status: str):
    """
    Retorna HTML para um estágio do pipeline.
    status: 'done', 'active', 'pending'
    """
    icons = {"done": "✓", "active": "●", "pending": "○"}
    icon = icons.get(status, "○")
    return f"""
    <div class="pipeline-stage {status}">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
    </div>
    """


def render_pipeline(stages: list):
    """
    stages: lista de tuplas (label, status)
    Exemplo: [("Catalog", "done"), ("Imagery", "done"), ("Spectral", "active")]
    """
    html = '<div class="pipeline">'
    for i, (label, status) in enumerate(stages):
        html += pipeline_stage(label, status)
        if i < len(stages) - 1:
            connector_class = "done" if status == "done" else ""
            html += f'<div class="pipeline-connector {connector_class}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def status_badge(text: str, type: str = "ready"):
    """Badge de status: ready, pending, error."""
    cls = f"status-badge status-{type}"
    return f'<span class="{cls}">{text}</span>'