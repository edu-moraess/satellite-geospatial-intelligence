"""
ui/components.py – Componentes reutilizáveis da UI.
"""

import streamlit as st

def section_title(title, description=""):
    """Título de seção com descrição opcional."""
    st.markdown(f'<div class="sgi-section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="sgi-section-description">{description}</div>', unsafe_allow_html=True)

def metric_card(label, value, change=None, icon=""):
    """Card de métrica compacto."""
    change_html = f'<span class="sgi-metric-change">{change}</span>' if change else ''
    icon_html = f'<span style="margin-right:4px;">{icon}</span>' if icon else ''
    st.markdown(f"""
    <div class="sgi-metric-card">
        <div class="sgi-metric-label">{icon_html}{label}</div>
        <div class="sgi-metric-value">{value}{change_html}</div>
    </div>
    """, unsafe_allow_html=True)

def pipeline_stage(name, state="pending"):
    """
    Exibe um estágio do pipeline.
    state: 'pending', 'active', 'done'
    """
    icons = {"pending": "○", "active": "●", "done": "✓"}
    colors = {"pending": "#5a667a", "active": "#e8edf5", "done": "#4caf50"}
    icon = icons.get(state, "○")
    color = colors.get(state, "#5a667a")
    st.markdown(f"""
    <span class="sgi-pipeline-stage {state}" style="color:{color};">
        <span class="status-icon">{icon}</span> {name}
    </span>
    """, unsafe_allow_html=True)

def scene_row(scene, download_callback):
    """
    Linha do catálogo de cenas.
    scene: dict com 'date', 'cloud', 'id', etc.
    download_callback: função chamada ao clicar no botão.
    """
    date = scene.get('date', '')
    cloud = scene.get('cloud', 0.0)
    scene_id = scene.get('id', '')[:8]
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.markdown(f'<span class="date">{date}</span>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<span class="cloud">{cloud:.2f}%</span>', unsafe_allow_html=True)
    with cols[2]:
        # Botão discreto
        if st.button("Download", key=f"dl_{scene_id}"):
            download_callback(scene)

def image_pair(label1, img1, label2, img2):
    """
    Exibe duas imagens lado a lado.
    """
    st.markdown('<div class="sgi-image-pair">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption=label1, use_container_width=True)
    with col2:
        st.image(img2, caption=label2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)