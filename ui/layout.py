"""
ui/layout.py – Funções para renderizar as seções principais do aplicativo.
Cada função recebe os dados necessários e desenha a UI correspondente.
"""

import streamlit as st
from ui.components import (
    metric_card,
    pipeline_stage,
    scene_row,
    image_pair,
    section_title
)
from ui.status import get_pipeline_status, update_pipeline_status

def render_header():
    """Cabeçalho global com título e status."""
    st.markdown("""
    <div class="sgi-header">
        <div class="sgi-brand">
            <div class="sgi-logo">🛰️ <span>SGI</span></div>
            <div>
                <div class="sgi-title">Satellite Geospatial Intelligence</div>
                <div class="sgi-subtitle">Earth Observation · Remote Sensing · Geospatial AI</div>
            </div>
        </div>
        <div class="sgi-status">
            <span class="sgi-status-dot"></span> SYSTEM ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_mission_summary(aoi_info, date_range, cloud_limit, scene_count):
    """Resumo compacto da missão atual."""
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        metric_card("AOI", aoi_info, icon="📍")
    with cols[1]:
        metric_card("Date Range", date_range, icon="📅")
    with cols[2]:
        metric_card("Cloud Limit", f"{cloud_limit}%", icon="☁️")
    with cols[3]:
        metric_card("Scenes", str(scene_count), icon="🛰️")

def render_geospatial_operations_center(map_func, aoi_type="polygon"):
    """
    Renderiza o mapa principal com seleção AOI.
    map_func: função que retorna o objeto folium e aceita parâmetros de estado.
    """
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Geospatial Operations Center", "Interactive Earth observation · Sentinel-2 · AOI · Spatial analysis")
    st.markdown('<div class="sgi-map-wrapper">', unsafe_allow_html=True)
    # Chama a função do mapa; ela deve usar st.session_state para persistência
    map_func()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_scene_catalog(scenes, download_callback):
    """
    Catálogo de cenas em formato de tabela compacta.
    scenes: lista de dicionários com 'date', 'cloud', 'id', etc.
    download_callback: função chamada ao clicar em download.
    """
    if not scenes:
        st.info("Nenhuma cena encontrada. Use a barra lateral para pesquisar.")
        return
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Satellite Archive", f"{len(scenes)} scenes available")
    st.markdown('<div class="sgi-scene-catalog">', unsafe_allow_html=True)
    for scene in scenes:
        scene_row(scene, download_callback)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_active_scene(scene_metadata, rgb_img, false_color_img):
    """
    Exibe metadados da cena ativa e as visualizações RGB e False Color.
    """
    if not scene_metadata:
        return
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Active Scene", f"{scene_metadata.get('date', '')} · Cloud {scene_metadata.get('cloud', 0):.2f}% · ID {scene_metadata.get('id', '')[:8]}")
    if rgb_img is not None and false_color_img is not None:
        image_pair("RGB", rgb_img, "False Color", false_color_img)
    elif rgb_img is not None:
        st.image(rgb_img, caption="RGB", use_container_width=True)
    elif false_color_img is not None:
        st.image(false_color_img, caption="False Color", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_spectral_intelligence(ndvi, ndwi, ndbi, spectral_map):
    """
    Exibe os índices espectrais e o mapa escolhido.
    """
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Spectral Intelligence", "NDVI · NDWI · NDBI")
    cols = st.columns(3)
    with cols[0]:
        metric_card("NDVI", f"{ndvi:.3f}" if ndvi is not None else "—")
    with cols[1]:
        metric_card("NDWI", f"{ndwi:.3f}" if ndwi is not None else "—")
    with cols[2]:
        metric_card("NDBI", f"{ndbi:.3f}" if ndbi is not None else "—")
    if spectral_map is not None:
        st.markdown('<div class="sgi-map-wrapper">', unsafe_allow_html=True)
        st.image(spectral_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_land_cover(classification_img, distribution_data, area_data):
    """
    Exibe classificação, distribuição e estimativa de área.
    """
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Land Cover", "Classification · Distribution · Area")
    if classification_img is not None:
        st.image(classification_img, use_container_width=True)
    if distribution_data:
        cols = st.columns(len(distribution_data))
        for col, (label, value) in zip(cols, distribution_data.items()):
            with col:
                metric_card(label, f"{value:.1f}%")
    if area_data:
        cols = st.columns(len(area_data))
        for col, (label, value) in zip(cols, area_data.items()):
            with col:
                metric_card(f"{label} (km²)", f"{value:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

def render_change_detection(before_img, after_img, change_index, sensitivity, change_stats, change_map):
    """
    Controles e visualização de detecção de mudanças.
    """
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Change Detection", "Compare two observations")
    # Controles em linha
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        st.selectbox("Before", options=["Scene A", "Scene B"], key="change_before")
    with col2:
        st.selectbox("After", options=["Scene B", "Scene A"], key="change_after")
    with col3:
        st.selectbox("Index", options=["NDVI", "NDWI", "NDBI"], key="change_index")
    with col4:
        st.slider("Sensitivity", 0.0, 1.0, 0.1, 0.05, key="change_sensitivity")
    if st.button("Analyze Changes", key="analyze_changes_btn"):
        # Chamada à função de detecção (deve ser definida em app.py)
        st.session_state['run_change_detection'] = True
    # Resultados
    if change_stats:
        cols = st.columns(3)
        with cols[0]:
            metric_card("Decrease", f"{change_stats.get('decrease', 0):.2f}%")
        with cols[1]:
            metric_card("Increase", f"{change_stats.get('increase', 0):.2f}%")
        with cols[2]:
            metric_card("Total Changed", f"{change_stats.get('total', 0):.2f}%")
    if change_map is not None:
        st.image(change_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_geospatial_ai(model_list, input_preview, config, detection_results, export_callback):
    """
    Módulo de IA geoespacial com configuração, execução e exportação.
    """
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    section_title("Geospatial AI", "Object Detection · Classification · Export")
    with st.expander("Model & Configuration", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.selectbox("Model", model_list, key="ai_model")
            st.text("Checkpoint: latest")
        with col2:
            if input_preview is not None:
                st.image(input_preview, caption="Input Preview", use_container_width=True)
        col3, col4, col5 = st.columns(3)
        with col3:
            st.number_input("Tile Size", 256, 1024, 512, 64, key="ai_tile_size")
        with col4:
            st.slider("Overlap", 0.0, 0.5, 0.2, 0.05, key="ai_overlap")
        with col5:
            st.slider("Confidence", 0.0, 1.0, 0.5, 0.05, key="ai_confidence")
        st.multiselect("Classes", options=["Vegetation", "Water", "Built-up", "Bare Soil", "Other"], key="ai_classes")
        if st.button("Run Geospatial AI", key="run_ai_btn"):
            st.session_state['run_ai'] = True
    if detection_results:
        st.metric("Objects Detected", detection_results.get('count', 0))
        st.metric("Classes", ", ".join(detection_results.get('classes', [])))
        if detection_results.get('image') is not None:
            st.image(detection_results['image'], use_container_width=True)
        if st.button("Export GeoJSON", key="export_geojson_btn"):
            export_callback()
    st.markdown('</div>', unsafe_allow_html=True)

def render_pipeline_status():
    """Barra horizontal de status do pipeline."""
    status = get_pipeline_status()
    stages = ["Catalog", "Imagery", "Spectral", "Change Detection", "Geospatial AI"]
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-pipeline">', unsafe_allow_html=True)
    for i, stage in enumerate(stages):
        state = status.get(stage, "pending")
        pipeline_stage(stage, state)
        if i < len(stages) - 1:
            st.markdown('<span class="sgi-pipeline-arrow">→</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_footer():
    """Rodapé compacto."""
    st.markdown("""
    <div class="sgi-footer">
        SATELLITE GEOSPATIAL INTELLIGENCE · Earth Observation · Remote Sensing · Computer Vision · Geospatial AI<br>
        Spectral values are analytical measurements and should be interpreted according to sensor characteristics, spatial resolution and preprocessing.
    </div>
    """, unsafe_allow_html=True)