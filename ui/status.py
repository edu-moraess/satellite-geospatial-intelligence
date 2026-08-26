"""
Indicadores de status e pipeline.
"""

import streamlit as st
from ui.theme import COLORS

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

def init_pipeline_status():
    """Inicializa o status do pipeline na sessão."""
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = {
            "Catalog": "pending",
            "Imagery": "pending",
            "Spectral": "pending",
            "Change": "pending",
            "AI": "pending",
        }


def update_pipeline_status(stage: str, status: str):
    """
    Atualiza o status de um estágio do pipeline.
    status: 'pending', 'active', 'done', 'error'
    """
    if "pipeline_status" in st.session_state:
        st.session_state.pipeline_status[stage] = status


def get_pipeline_status():
    """Retorna o dicionário de status do pipeline."""
    if "pipeline_status" not in st.session_state:
        init_pipeline_status()
    return st.session_state.pipeline_status


# =============================================================================
# RENDERIZAÇÃO DO PIPELINE
# =============================================================================

def render_pipeline_status(stages: dict):
    """
    stages: dict com chaves = nome do estágio, valor = 'done', 'active', 'pending', 'error'
    """
    order = ["Catalog", "Imagery", "Spectral", "Change", "AI"]

    html = '<div class="pipeline">'
    for i, name in enumerate(order):
        status = stages.get(name, "pending")

        # Normaliza para os estados do CSS
        if status == "error":
            css_status = "error"
            icon = "✗"
        elif status == "done":
            css_status = "done"
            icon = "✓"
        elif status == "active":
            css_status = "active"
            icon = "●"
        else:
            css_status = "pending"
            icon = "○"

        html += f"""
        <div class="pipeline-stage {css_status}">
            <div class="icon">{icon}</div>
            <div class="label">{name}</div>
        </div>
        """
        if i < len(order) - 1:
            connector_class = "done" if stages.get(order[i]) == "done" else ""
            html += f'<div class="pipeline-connector {connector_class}"></div>'
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# STATUS INDICATOR
# =============================================================================

def status_indicator(text: str, type: str = "active"):
    """
    type: 'active', 'ready', 'pending', 'error'
    """
    colors = {
        "active": COLORS["accent"],
        "ready": COLORS["success"],
        "pending": COLORS["text_secondary"],
        "error": COLORS["danger"],
    }
    color = colors.get(type, COLORS["text_secondary"])
    return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};margin-right:0.4rem;"></span><span style="font-size:0.7rem;color:{COLORS["text_secondary"]};text-transform:uppercase;letter-spacing:0.04em;">{text}</span>'