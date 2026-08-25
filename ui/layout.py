"""
ui/layout.py – Funções de renderização das seções principais.
Ajustado para criar colunas dinamicamente e evitar IndexError.
"""

import streamlit as st
from ui.components import metric_card, pipeline_stage
from ui.status import get_pipeline_status

def render_header():
    """Cabeçalho global."""
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

def render_mission_summary(items, drawn_aoi, latitude, longitude, area_size):
    """KPIs compactos da missão."""
    if items:
        sorted_items = sorted(items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100)))
        best_cloud = float(sorted_items[0].properties.get("eo:cloud_cover", 0))
        latest_dates = [item.datetime.date() for item in items if item.datetime]
        latest_date = max(latest_dates) if latest_dates else "N/A"
    else:
        best_cloud = 0.0
        latest_date = "N/A"

    cols = st.columns(4)
    with cols[0]:
        metric_card("Scenes", str(len(items)) if items else "0")
    with cols[1]:
        metric_card("Best Cloud", f"{best_cloud:.2f}%")
    with cols[2]:
        metric_card("Latest", str(latest_date))
    with cols[3]:
        metric_card("AOI", "Drawn" if drawn_aoi else f"{latitude:.4f}° / {longitude:.4f}°")

def render_geospatial_operations_center(map_panel_func):
    """Renderiza o mapa principal."""
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Geospatial Operations Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-description">Interactive Earth observation · Sentinel-2 · AOI · Spatial analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-map-wrapper">', unsafe_allow_html=True)
    map_panel_func()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_scene_catalog(items, download_callback):
    """Catálogo de cenas em tabela compacta."""
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Satellite Archive</div>', unsafe_allow_html=True)
    if not items:
        st.info("Search the Sentinel-2 catalog from the sidebar to populate the archive.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown(f'<div class="sgi-section-description">{len(items)} scenes available</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-scene-catalog">', unsafe_allow_html=True)
    for idx, item in enumerate(items):
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        date_str = str(item.datetime.date()) if item.datetime else "Unknown"
        quality = "Excellent" if cloud <= 1 else "Good" if cloud <= 5 else "Acceptable" if cloud <= 10 else "Cloudy"
        cols = st.columns([2, 1, 1, 1])
        with cols[0]:
            st.markdown(f'<span class="date-col">{date_str}</span>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<span class="cloud-col">{cloud:.2f}%</span>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<span class="quality-col">{quality}</span>', unsafe_allow_html=True)
        with cols[3]:
            if st.button("Download", key=f"dl_{idx}_{item.id[:8]}"):
                download_callback(item)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_active_scene(data, rgb, false_color):
    """Exibe metadados e visualizações da cena ativa."""
    if data is None:
        st.info("Download a satellite scene to activate analysis.")
        return
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Active Scene</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sgi-section-description">{data["date"]} · Cloud {data["cloud"]:.2f}% · ID {data["scene_id"][:16]}</div>', unsafe_allow_html=True)
    if rgb is not None and false_color is not None:
        st.markdown('<div class="sgi-image-pair">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb, caption="Natural Color", use_container_width=True)
        with col2:
            st.image(false_color, caption="False Color", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif rgb is not None:
        st.image(rgb, caption="Natural Color", use_container_width=True)
    elif false_color is not None:
        st.image(false_color, caption="False Color", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_spectral_intelligence(ndvi, ndwi, ndbi, index_figure):
    """Índices espectrais e mapa de índice."""
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Spectral Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-description">NDVI · NDWI · NDBI</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        metric_card("NDVI", f"{ndvi:.3f}" if ndvi is not None else "—")
    with cols[1]:
        metric_card("NDWI", f"{ndwi:.3f}" if ndwi is not None else "—")
    with cols[2]:
        metric_card("NDBI", f"{ndbi:.3f}" if ndbi is not None else "—")
    if index_figure is not None:
        st.pyplot(index_figure, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_land_cover(classification, percentages, area):
    """Classificação e estatísticas de cobertura do solo – CORRIGIDO: colunas dinâmicas."""
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Land Cover</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-description">Rule‑based baseline</div>', unsafe_allow_html=True)
    if classification is not None:
        st.pyplot(classification, use_container_width=True)
        # Percentagens – colunas dinâmicas
        if percentages:
            cols = st.columns(len(percentages))
            for i, (label, pct) in enumerate(percentages.items()):
                with cols[i]:
                    metric_card(label, f"{pct:.1f}%")
        # Áreas – colunas dinâmicas
        if area:
            cols = st.columns(len(area))
            for i, (label, val) in enumerate(area.items()):
                with cols[i]:
                    metric_card(f"{label} km²", f"{val:.3f}")
    else:
        st.info("No classification available.")
    st.markdown('</div>', unsafe_allow_html=True)

def render_change_detection_controls(items, drawn_aoi, latitude, longitude, area_size):
    """Controles e execução de change detection, exibe resultados se disponíveis."""
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Change Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-description">Compare two observations</div>', unsafe_allow_html=True)

    if len(items) < 2:
        st.info("Search for at least two scenes to enable change detection.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    scene_options = {}
    for item in items:
        date_str = str(item.datetime.date()) if item.datetime else "Unknown"
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        label = f"{date_str} • {cloud:.2f}% clouds • {item.id[:8]}"
        scene_options[label] = item
    scene_names = list(scene_options.keys())

    col1, col2 = st.columns(2)
    with col1:
        before_name = st.selectbox("Data A — Before", scene_names, key="change_before")
    with col2:
        after_name = st.selectbox("Data B — After", scene_names, index=min(1, len(scene_names)-1), key="change_after")

    col3, col4 = st.columns(2)
    with col3:
        threshold = st.slider("Sensitivity", 0.01, 0.50, 0.10, 0.01, key="change_threshold")
    with col4:
        index_choice = st.selectbox("Index", ["NDVI — Vegetation", "NDWI — Water", "NDBI — Built-up"], key="change_index")

    if st.button("Analyze Changes", type="primary", key="run_change_detection"):
        st.session_state['run_change_detection'] = True

    # Resultados (se existirem)
    change_result = st.session_state.get('change_result')
    if change_result:
        stats = change_result['statistics']
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Decrease", f"{stats['decrease_km2']:.3f} km²")
        with c2:
            metric_card("Increase", f"{stats['increase_km2']:.3f} km²")
        with c3:
            metric_card("Total Changed", f"{stats['total_changed_km2']:.3f} km²")
        if change_result.get('figure'):
            st.pyplot(change_result['figure'], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_geospatial_ai(data, detection_rgb, detections):
    """Seção de IA geoespacial."""
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Geospatial AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-description">Object Detection · Classification · Export</div>', unsafe_allow_html=True)

    if data is None or detection_rgb is None:
        st.info("Download a scene to enable Geospatial AI.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    with st.expander("Model & Configuration", expanded=False):
        st.image(detection_rgb, caption="Input RGB", use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            tile_size = st.selectbox("Tile size", [256, 512, 768, 1024], index=1, key="ai_tile_size")
            overlap = st.slider("Overlap", 0, 256, 64, 16, key="ai_overlap")
        with col2:
            confidence = st.slider("Confidence", 0.10, 0.95, 0.50, 0.05, key="ai_confidence")
            try:
                from src.model_registry import list_models
                model_ids = list_models()
                selected_model = st.selectbox("Model", model_ids, key="ai_model")
            except:
                selected_model = None
            classes = st.multiselect("Classes", ["Vegetation", "Water", "Built-up", "Bare Soil", "Other"], key="ai_classes")
        if st.button("Run Geospatial AI", type="primary", key="run_ai"):
            st.session_state['run_ai'] = True

    if detections:
        try:
            from src.object_detection import detection_summary
            summary = detection_summary(detections)
        except:
            summary = {}
        col1, col2 = st.columns(2)
        with col1:
            metric_card("Objects", str(len(detections)))
        with col2:
            metric_card("Classes", str(len(summary)))
        detection_fig = st.session_state.get('detection_figure')
        if detection_fig:
            st.pyplot(detection_fig, use_container_width=True)
        if st.button("Export GeoJSON", key="export_geojson"):
            st.session_state['export_geojson'] = True
    else:
        st.info("No detections available. Run inference to generate results.")
    st.markdown('</div>', unsafe_allow_html=True)

def render_pipeline_status():
    """Barra horizontal de status do pipeline."""
    status = get_pipeline_status()
    stages = ["Catalog", "Imagery", "Spectral", "Change", "AI"]
    st.markdown('<div class="sgi-section">', unsafe_allow_html=True)
    st.markdown('<div class="sgi-section-title">Processing Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sgi-pipeline">', unsafe_allow_html=True)
    for i, stage in enumerate(stages):
        state = status.get(stage, "pending")
        pipeline_stage(stage, state)
        if i < len(stages)-1:
            st.markdown('<span class="sgi-pipeline-arrow">→</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_footer():
    """Rodapé."""
    st.markdown("""
    <div class="sgi-footer">
        SATELLITE GEOSPATIAL INTELLIGENCE · Earth Observation · Remote Sensing · Computer Vision · Geospatial AI<br>
        Spectral values are analytical measurements and should be interpreted according to sensor characteristics, spatial resolution and preprocessing.
    </div>
    """, unsafe_allow_html=True)