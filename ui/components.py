"""
ui/components.py – Componentes reutilizáveis.
"""

import streamlit as st

def metric_card(label, value):
    """Card de métrica compacto."""
    st.markdown(f"""
    <div class="sgi-metric-card">
        <div class="sgi-metric-label">{label}</div>
        <div class="sgi-metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def pipeline_stage(name, state="pending"):
    """Estágio do pipeline."""
    icons = {"pending": "○", "active": "●", "done": "✓"}
    colors = {"pending": "#5a667a", "active": "#e8edf5", "done": "#4caf50"}
    icon = icons.get(state, "○")
    color = colors.get(state, "#5a667a")
    st.markdown(f"""
    <span class="sgi-pipeline-stage {state}" style="color:{color};">
        <span class="status-icon">{icon}</span> {name}
    </span>
    """, unsafe_allow_html=True)