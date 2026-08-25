"""
ui/status.py – Gerenciamento do estado do pipeline.
Utiliza st.session_state para persistência.
"""

import streamlit as st

PIPELINE_STAGES = ["Catalog", "Imagery", "Spectral", "Change Detection", "Geospatial AI"]

def init_pipeline_status():
    """Inicializa o status do pipeline no session_state."""
    if "pipeline_status" not in st.session_state:
        st.session_state["pipeline_status"] = {stage: "pending" for stage in PIPELINE_STAGES}

def get_pipeline_status():
    """Retorna o dicionário de status atual."""
    init_pipeline_status()
    return st.session_state["pipeline_status"]

def update_pipeline_status(stage, state):
    """
    Atualiza o status de um estágio.
    state: 'pending', 'active', 'done'
    """
    init_pipeline_status()
    if stage in st.session_state["pipeline_status"]:
        st.session_state["pipeline_status"][stage] = state
    else:
        st.warning(f"Estágio '{stage}' não reconhecido.")

def reset_pipeline():
    """Reinicia todos os estágios para 'pending'."""
    init_pipeline_status()
    for stage in PIPELINE_STAGES:
        st.session_state["pipeline_status"][stage] = "pending"