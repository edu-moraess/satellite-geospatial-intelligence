"""
Componentes reutilizáveis da interface.
Cards, tabelas, botões especiais, etc.
"""

import streamlit as st
from ui.theme import COLORS
from ui.layout import status_badge

# =============================================================================
# HEADER
# =============================================================================

def render_header():
    """Header compacto com título e status do sistema."""
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0 1rem 0;
            border-bottom: 1px solid {COLORS["border"]};
            margin-bottom: 1.5rem;
        ">
            <div>
                <span style="font-size:1.3rem;font-weight:600;letter-spacing:-0.02em;color:{COLORS["text_primary"]};">
                    GEOCORE
                </span>
                <div style="font-size:0.75rem;color:{COLORS["text_secondary"]};margin-top:0.1rem;">
                    Geospatial Computing & Intelligence
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{COLORS["success"]};"></span>
                <span style="font-size:0.7rem;color:{COLORS["text_secondary"]};text-transform:uppercase;letter-spacing:0.06em;">
                    SYSTEM ONLINE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MISSION CONTROL (resumo na página principal)
# =============================================================================

def render_mission_control(aoi_data: dict):
    """
    Painel compacto com resumo da missão.
    aoi_data: dict com lat, lon, area, time_window, cloud_coverage, scenes
    """
    cols = st.columns(6)
    metrics = [
        ("AOI", f"{aoi_data.get('lat', '—')}, {aoi_data.get('lon', '—')}"),
        ("Area", aoi_data.get('area', '—')),
        ("Time Window", aoi_data.get('time_window', '—')),
        ("Cloud Coverage", aoi_data.get('cloud_coverage', '—')),
        ("Scenes", str(aoi_data.get('scenes', 0))),
        ("Status", status_badge("Active", "ready")),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div style="padding:0.2rem 0;">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:{COLORS["text_secondary"]};">{label}</div>
                    <div style="font-size:0.95rem;font-weight:500;color:{COLORS["text_primary"]};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# SPECTRAL INTELLIGENCE – Cards de índices
# =============================================================================

def render_spectral_cards(indices: dict):
    """
    indices: dict com 'NDVI', 'NDWI', 'NDBI' e seus valores
    """
    cols = st.columns(3)
    for col, (name, value) in zip(cols, indices.items()):
        with col:
            st.markdown(
                f"""
                <div class="panel" style="text-align:center;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:{COLORS["text_secondary"]};">{name}</div>
                    <div style="font-size:1.8rem;font-weight:600;color:{COLORS["text_primary"]};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# CHANGE INTELLIGENCE – Métricas de mudança
# =============================================================================

def render_change_metrics(decrease: str, increase: str, total: str):
    cols = st.columns(3)
    metrics = [
        ("DECREASE", decrease, COLORS["danger"]),
        ("INCREASE", increase, COLORS["success"]),
        ("TOTAL CHANGED", total, COLORS["accent"]),
    ]
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="panel" style="text-align:center;padding:0.6rem 0.5rem;">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:{COLORS["text_secondary"]};">{label}</div>
                    <div style="font-size:1.6rem;font-weight:600;color:{color};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# GEOSPATIAL AI – Configuração e resultados
# =============================================================================

def render_ai_config():
    """Exibe controles de configuração do modelo em um layout compacto."""
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05)
    with col2:
        tile_size = st.selectbox("Tile Size", [256, 512, 1024], index=1)
    with col3:
        overlap = st.slider("Overlap", 0.0, 0.5, 0.2, 0.05)
    classes = st.multiselect(
        "Classes",
        ["Vegetation", "Water", "Built-up", "Bare Soil", "Other"],
        default=["Vegetation", "Built-up"],
    )
    return confidence, tile_size, overlap, classes 