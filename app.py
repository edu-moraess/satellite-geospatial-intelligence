from __future__ import annotations
from datetime import date
import numpy as np
import streamlit as st

from src.catalog import search_sentinel, create_bbox
from src.config import RAW_DIR
from src.downloader import download_required_bands
from src.geospatial import read_band, align_band_to_reference
from src.visualization import create_rgb, create_false_color
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi
from src.index_visualization import create_index_figure
from src.classification import classify_land_cover, calculate_class_percentages
from src.land_cover import create_land_cover_figure, calculate_area_km2
from src.change_detection import calculate_difference, detect_change, calculate_change_statistics
from src.change_visualization import create_change_figure
from src.object_detection import normalize_rgb, validate_detection_image, filter_detections, filter_classes, detection_summary, draw_detections
from src.tiling import create_tiles, tile_count
from src.detector_model import SatelliteDetector
from src.model_registry import list_models, get_model
from src.map_view import render_map_panel

# ============================================================
# CONFIGURAÇÃO DA PÁGINA & CSS
# ============================================================
st.set_page_config(page_title="Satellite Geospatial Intelligence", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    .stApp { background-color: #0a0f16; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    h1, h2, h3 { color: #ffffff !important; border-bottom: 1px solid #1f2937; padding-bottom: 10px; text-shadow: 0 0 10px rgba(0,212,255,0.3); }
    .stTabs [data-baseweb="tab-list"] { background-color: #111827; border-radius: 8px; padding: 4px; border: 1px solid #1f2937; }
    .stTabs [data-baseweb="tab"] { color: #9ca3af; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background-color: #1f2937; color: #00d4ff !important; }
    [data-testid="stMetric"] { background: rgba(17, 24, 39, 0.8); backdrop-filter: blur(10px); border: 1px solid #1f2937; border-radius: 10px; padding: 15px; transition: all 0.3s; }
    [data-testid="stMetric"]:hover { border-color: #00d4ff; box-shadow: 0 0 15px rgba(0, 212, 255, 0.2); transform: translateY(-2px); }
    [data-testid="stMetricLabel"] { color: #9ca3af; text-transform: uppercase; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #ffffff; font-weight: 700; font-size: 1.5rem; }
    .stButton > button { background-color: #1f2937; color: #e0e0e0; border: 1px solid #374151; border-radius: 6px; }
    .stButton > button[kind="primary"] { background: linear-gradient(90deg, #00b4d8, #0077b6); color: #fff; border: none; }
    div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input { background-color: #1f2937; border-color: #374151; color: white; }
    .stAlert { background-color: #1f2937; border-left: 4px solid #00d4ff; border-radius: 6px; }
    .stImage img, .stPlotlyChart, .stPydeckChart { border-radius: 10px; border: 1px solid #1f2937; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    .streamlit-expanderHeader { background-color: #1f2937; color: #00d4ff !important; border-radius: 8px; border: 1px solid #1f2937; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ESTADO DA SESSÃO
# ============================================================
DEFAULT_STATE = {"search_results": [], "satellite_data": None, "change_result": None, "object_detections": [], "map_state": {}}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state: st.session_state[key] = value

def align_arrays_for_change(before, after):
    before = np.asarray(before); after = np.asarray(after)
    if before.ndim != 2 or after.ndim != 2: raise ValueError("Change detection requires 2D arrays.")
    h = min(before.shape[0], after.shape[0]); w = min(before.shape[1], after.shape[1])
    if h <= 0 or w <= 0: raise ValueError("Invalid array dimensions after alignment.")
    return before[:h, :w], after[:h, :w]

# ============================================================
# CABEÇALHO E SIDEBAR
# ============================================================
st.title("🛰️ Satellite Geospatial Intelligence")
st.caption("Earth Observation • Computer Vision • Geospatial AI")

st.sidebar.header("📍 Area of Interest")
latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, -23.5505, format="%.6f")
longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, -46.6333, format="%.6f")
area_size = st.sidebar.slider("Area size (degrees)", 0.01, 0.20, 0.05, 0.01)
st.sidebar.header("📅 Satellite Date Range")
start_date = st.sidebar.date_input("Start date", date(2026, 1, 1))
end_date = st.sidebar.date_input("End date", date(2026, 8, 23))
st.sidebar.header("☁️ Image Quality")
max_cloud_cover = st.sidebar.slider("Maximum cloud coverage", 0, 100, 10, 1, format="%d%%")

if st.sidebar.button("🔎 Search Satellite Data", type="primary", width='stretch'):
    if start_date > end_date: st.error("❌ Start date must be before the end date."); st.stop()
    with st.spinner("🛰️ Searching Sentinel‑2 catalog..."):
        try:
            results = search_sentinel(latitude, longitude, area_size, str(start_date), str(end_date), max_cloud_cover)
            st.session_state.search_results = results
            st.session_state.satellite_data = None
            st.session_state.change_result = None
            st.session_state.object_detections = []
            st.rerun()
        except Exception as error:
            st.error("❌ Satellite catalog search failed.")
            with st.expander("Technical details"): st.exception(error)
            st.stop()

items = st.session_state.search_results
data = st.session_state.satellite_data

# ============================================================
# LISTA DE CENAS OCULTA (EXPANDER FECHADO)
# ============================================================
if items:
    st.success(f"🛰️ {len(items)} satellite scenes found.")
    
    with st.expander("📡 Available Sentinel-2 Scenes", expanded=False):
        st.caption("Selecione uma cena para fazer o download e iniciar a análise.")
        for index, item in enumerate(items[:20]):
            cloud = float(item.properties.get("eo:cloud_cover", 0))
            acquisition_date = item.datetime.date() if item.datetime else "Unknown"
            
            with st.expander(f"{acquisition_date} • {cloud:.2f}% clouds"):
                st.write(f"**Scene ID:** `{item.id}`")
                if st.button("⬇️ Download & Analyze", key=f"download_{index}", width='stretch'):
                    bbox = create_bbox(latitude, longitude, area_size)
                    output_dir = RAW_DIR / item.id
                    with st.spinner("⬇️ Downloading satellite data..."):
                        try:
                            downloaded = download_required_bands(item, bbox, output_dir)
                        except Exception as error:
                            st.error("❌ Satellite download failed.")
                            with st.expander("Technical details"): st.exception(error)
                            st.stop()
                            
                    st.session_state.satellite_data = {
                        "scene_id": item.id, "date": str(acquisition_date), "cloud": cloud,
                        "bands": downloaded, "latitude": latitude, "longitude": longitude, "area_size": area_size
                    }
                    st.session_state.change_result = None
                    st.session_state.object_detections = []
                    st.success("✅ Satellite scene downloaded. Iniciando análise...")
                    st.rerun()

data = st.session_state.satellite_data
detection_rgb = None

# ============================================================
# PROCESSAMENTO PRINCIPAL (ANTES DAS ABAS)
# ============================================================
if data:
    with st.spinner("📡 Loading spectral bands..."):
        try:
            b02, m02 = read_band(data["bands"]["B02"])
            b03, m03 = read_band(data["bands"]["B03"])
            b04, m04 = read_band(data["bands"]["B04"])
            b08, m08 = read_band(data["bands"]["B08"])
            b11, m11 = read_band(data["bands"]["B11"])
        except Exception as error:
            st.error("❌ Failed to load satellite bands.")
            with st.expander("Technical details"): st.exception(error)
            st.stop()

    with st.spinner("🔄 Aligning spectral grids..."):
        try:
            b02 = align_band_to_reference(b02, m02, b04, m04)
            b03 = align_band_to_reference(b03, m03, b04, m04)
            b08 = align_band_to_reference(b08, m08, b04, m04)
            b11 = align_band_to_reference(b11, m11, b04, m04)
        except Exception as error:
            st.error("❌ Failed to align spectral bands.")
            with st.expander("Technical details"): st.exception(error)
            st.stop()

    try:
        rgb = create_rgb(blue=b02, green=b03, red=b04)
        detection_rgb = normalize_rgb(red=b04, green=b03, blue=b02)
        validate_detection_image(detection_rgb)
        false_color = create_false_color(green=b03, red=b04, nir=b08)
        ndvi = calculate_ndvi(red=b04, nir=b08)
        ndwi = calculate_ndwi(green=b03, nir=b08)
        ndbi = calculate_ndbi(nir=b08, swir=b11)
        classification = classify_land_cover(ndvi=ndvi, ndwi=ndwi, ndbi=ndbi)
    except Exception as error:
        st.error("❌ Failed to create images.")
        with st.expander("Technical details"): st.exception(error)
        st.stop()

# ============================================================
# ABAS DO DASHBOARD
# ============================================================
if data:
    tab1, tab2, tab3, tab4 = st.tabs(["🛰️ Operations Center", "🔬 Spectral Analysis", "🌍 Change Detection", "🎯 Geospatial AI"])

    # ---------------- TAB 1: OPERATIONS CENTER ----------------
    with tab1:
        st.header("🛰️ Selected Satellite Scene")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Acquisition", data["date"])
        with c2: st.metric("Cloud Coverage", f"{data['cloud']:.2f}%")
        with c3: st.metric("Scene", data["scene_id"][:24])

        st.divider()
        
        map_col, controls_col = st.columns([3, 1])
        with map_col:
            try:
                scene_bbox = create_bbox(data.get("latitude", latitude), data.get("longitude", longitude), data.get("area_size", area_size))
                detections = st.session_state.object_detections
                
                render_map_panel(
                    latitude=data.get("latitude", latitude), longitude=data.get("longitude", longitude),
                    area_size=data.get("area_size", area_size), bbox=scene_bbox,
                    scene_id=data["scene_id"], acquisition_date=data["date"],
                    cloud_cover=data["cloud"], 
                    classification=classification, 
                    ndvi=ndvi, 
                    detections=detections if len(detections) > 0 else None,
                    key="main_geospatial_map"
                )
            except Exception as error:
                st.warning("⚠️ Interactive geospatial map could not be rendered.")
                with st.expander("Technical details"): st.exception(error)
        
        with controls_col:
            st.subheader("Layers")
            st.caption("Camadas de Dados Ativas no Mapa:")
            st.success("🟩 NDVI (Vegetação)")
            st.success("🟥 Classificação do Solo")
            if len(detections) > 0:
                st.success("🟪 Detecções IA")

    # ---------------- TAB 2: SPECTRAL ANALYSIS ----------------
    with tab2:
        col1, col2 = st.columns(2)
        with col1: st.subheader("🌍 Natural RGB"); st.image(rgb, width='stretch')
        with col2: st.subheader("🌱 False Color"); st.image(false_color, width='stretch')

        st.divider()
        st.header("🔬 Multispectral Indices")
        valid_ndvi = ndvi[np.isfinite(ndvi)]
        valid_ndwi = ndwi[np.isfinite(ndwi)]
        valid_ndbi = ndbi[np.isfinite(ndbi)]

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("🌱 Mean NDVI", f"{np.mean(valid_ndvi):.3f}" if valid_ndvi.size else "N/A")
        with m2: st.metric("💧 Mean NDWI", f"{np.mean(valid_ndwi):.3f}" if valid_ndwi.size else "N/A")
        with m3: st.metric("🏙️ Mean NDBI", f"{np.mean(valid_ndbi):.3f}" if valid_ndbi.size else "N/A")

        st.subheader("🗺️ Land Cover Classification")
        land_cover_figure = create_land_cover_figure(classification)
        st.plotly_chart(land_cover_figure, width='stretch')

        percentages = calculate_class_percentages(classification)
        cols = st.columns(5)
        with cols[0]: st.metric("🌳 Vegetation", f"{percentages['Vegetation']:.1f}%")
        with cols[1]: st.metric("💧 Water", f"{percentages['Water']:.1f}%")
        with cols[2]: st.metric("🏙️ Built‑up", f"{percentages['Built-up']:.1f}%")
        with cols[3]: st.metric("🟫 Bare Soil", f"{percentages['Bare Soil']:.1f}%")
        with cols[4]: st.metric("⬜ Other", f"{percentages['Other']:.1f}%")
        
        area = calculate_area_km2(classification, pixel_size_meters=10.0)
        st.subheader("📐 Estimated Area")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("🌳 Vegetation", f"{area['Vegetation']:.3f} km²")
        with c2: st.metric("💧 Water", f"{area['Water']:.3f} km²")
        with c3: st.metric("🏙️ Built‑up", f"{area['Built-up']:.3f} km²")
        with c4: st.metric("🟫 Bare Soil", f"{area['Bare Soil']:.3f} km²")

        st.divider()
        st.header("🔬 Spectral Index Maps")
        selected_index = st.selectbox("Choose index", ["NDVI — Vegetation", "NDWI — Water", "NDBI — Built‑up"], key="main_index")
        if selected_index.startswith("NDVI"): index_data, index_title, index_cmap = ndvi, "NDVI — Vegetation", "RdYlGn"
        elif selected_index.startswith("NDWI"): index_data, index_title, index_cmap = ndwi, "NDWI — Water", "Blues"
        else: index_data, index_title, index_cmap = ndbi, "NDBI — Built‑up", "Oranges"
        
        fig = create_index_figure(index_data, index_title, cmap=index_cmap)
        st.plotly_chart(fig, width='stretch')

    # ---------------- TAB 3: CHANGE DETECTION ----------------
    with tab3:
        st.header("🛰️ Change Detection")
        if len(items) >= 2:
            scene_map = {}
            for item in items:
                scene_date = item.datetime.date() if item.datetime else "Unknown"
                cloud = float(item.properties.get("eo:cloud_cover", 0))
                label = f"{scene_date} • {cloud:.2f}% clouds"
                scene_map[label] = item
            labels = list(scene_map.keys())
            col1, col2 = st.columns(2)
            with col1: before_label = st.selectbox("📅 Data A — Before", labels, key="change_before")
            with col2: after_label = st.selectbox("📅 Data B — After", labels, index=min(1, len(labels) - 1), key="change_after")

            threshold = st.slider("🎚️ Change sensitivity", 0.01, 0.50, 0.10, 0.01, key="change_threshold")
            change_index_choice = st.selectbox("🔬 Index to compare", ["NDVI — Vegetation", "NDWI — Water", "NDBI — Built‑up"], key="change_index")

            if st.button("🔍 Analyze Changes", type="primary", width='stretch'):
                try:
                    before_item = scene_map[before_label]
                    after_item = scene_map[after_label]
                    if before_item.id == after_item.id:
                        st.warning("⚠️ Please choose two different scenes.")
                    else:
                        bbox = create_bbox(latitude, longitude, area_size)
                        with st.spinner("🛰️ Downloading Data A..."): before_bands = download_required_bands(before_item, bbox, RAW_DIR / before_item.id)
                        with st.spinner("🛰️ Downloading Data B..."): after_bands = download_required_bands(after_item, bbox, RAW_DIR / after_item.id)
                        b04_before, m04_before = read_band(before_bands["B04"])
                        b03_before, m03_before = read_band(before_bands["B03"])
                        b08_before, m08_before = read_band(before_bands["B08"])
                        b11_before, m11_before = read_band(before_bands["B11"])
                        b04_after, m04_after = read_band(after_bands["B04"])
                        b03_after, m03_after = read_band(after_bands["B03"])
                        b08_after, m08_after = read_band(after_bands["B08"])
                        b11_after, m11_after = read_band(after_bands["B11"])

                        b03_before = align_band_to_reference(b03_before, m03_before, b04_before, m04_before)
                        b08_before = align_band_to_reference(b08_before, m08_before, b04_before, m04_before)
                        b11_before = align_band_to_reference(b11_before, m11_before, b04_before, m04_before)
                        b03_after = align_band_to_reference(b03_after, m03_after, b04_after, m04_after)
                        b08_after = align_band_to_reference(b08_after, m08_after, b04_after, m04_after)
                        b11_after = align_band_to_reference(b11_after, m11_after, b04_after, m04_after)

                        if change_index_choice.startswith("NDVI"):
                            before_index = calculate_ndvi(b04_before, b08_before)
                            after_index = calculate_ndvi(b04_after, b08_after)
                            index_name = "NDVI — Vegetation"
                        elif change_index_choice.startswith("NDWI"):
                            before_index = calculate_ndwi(b03_before, b08_before)
                            after_index = calculate_ndwi(b03_after, b08_after)
                            index_name = "NDWI — Water"
                        else:
                            before_index = calculate_ndbi(b08_before, b11_before)
                            after_index = calculate_ndbi(b08_after, b11_after)
                            index_name = "NDBI — Built‑up"

                        before_index, after_index = align_arrays_for_change(before_index, after_index)
                        difference = calculate_difference(before_index, after_index)
                        change_map = detect_change(difference, threshold=threshold)
                        statistics = calculate_change_statistics(change_map, pixel_size_meters=10.0)

                        st.session_state.change_result = {
                            "difference": difference, "change_map": change_map, "statistics": statistics,
                            "index_name": index_name, "before_id": before_item.id, "after_id": after_item.id
                        }
                        st.success("✅ Change detection completed.")
                except Exception as error:
                    st.error("❌ Change detection could not be completed.")
                    with st.expander("Technical details"): st.exception(error)
        else:
            st.info("ℹ️ Search for at least two satellite scenes to activate Change Detection.")

        change_result = st.session_state.change_result
        if change_result:
            st.subheader(f"📊 {change_result['index_name']}")
            stats = change_result["statistics"]
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🔴 Decrease", f"{stats['decrease_km2']:.3f} km²")
            with c2: st.metric("🟢 Increase", f"{stats['increase_km2']:.3f} km²")
            with c3: st.metric("🛰️ Total Changed", f"{stats['total_changed_km2']:.3f} km²")
            fig = create_change_figure(change_result["change_map"], title=f"{change_result['index_name']} Change Detection")
            st.plotly_chart(fig, width='stretch')

    # ---------------- TAB 4: GEOSPATIAL AI ----------------
    with tab4:
        st.header("🎯 Geospatial AI")
        if data is None: st.info("ℹ️ Download a satellite scene to activate Geospatial AI.")
        elif detection_rgb is None: st.warning("⚠️ RGB image is unavailable for AI.")
        else:
            st.image(detection_rgb, caption="Sentinel‑2 RGB prepared for Geospatial AI", width='stretch')
            col1, col2 = st.columns(2)
            with col1: tile_size = st.selectbox("Tile size", [256, 512, 768, 1024], index=1, key="tile_size")
            with col2: tile_overlap = st.slider("Tile overlap", 0, 256, 64, 16, key="tile_overlap")
            confidence_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05, key="object_confidence")

            try:
                model_ids = list_models()
                selected_model_id = st.selectbox("Model", model_ids, key="selected_model")
                detector = SatelliteDetector(model_id=selected_model_id, device="cpu")
                model_info = detector.info()
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Model", model_info["model"])
                with c2: st.metric("Input", f"{model_info['input_size']}×{model_info['input_size']}")
                with c3:
                    if model_info["checkpoint_available"]: st.success("CHECKPOINT FOUND")
                    else: st.warning("CHECKPOINT MISSING")
                
                detection_classes = st.multiselect("Classes of interest", list(model_info["classes"]), default=list(model_info["classes"][:2]), key="object_classes")

                if st.button("🤖 Run Geospatial AI", type="primary", width='stretch'):
                    if not detection_classes: st.warning("⚠️ Select at least one class.")
                    elif not model_info["checkpoint_available"]: st.info("🧠 Model pipeline is ready, but the selected checkpoint is not installed yet.")
                    else:
                        try:
                            with st.spinner("🧠 Running Geospatial AI..."):
                                tiles = create_tiles(detection_rgb, tile_size=tile_size, overlap=tile_overlap)
                                detections = detector.predict_tiles(tiles, confidence=confidence_threshold)
                                detections = filter_detections(detections, confidence_threshold)
                                detections = filter_classes(detections, detection_classes)
                                st.session_state.object_detections = detections
                            st.success(f"✅ {len(tiles)} tiles processed. Volte à aba Operations Center para ver as caixas no mapa.")
                        except Exception as error:
                            st.error("❌ AI inference failed.")
                            with st.expander("Technical details"): st.exception(error)

                detections = st.session_state.object_detections
                if detections:
                    summary = detection_summary(detections)
                    st.subheader("📊 Detection Results")
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Objects", len(detections))
                    with c2: st.metric("Classes", len(summary))
                    fig = draw_detections(detection_rgb, detections)
                    st.pyplot(fig, width='stretch')

            except Exception as error:
                st.error("❌ Model registry failed.")
                with st.expander("Technical details"): st.exception(error)

else:
    st.info("🛰️ Procure e baixe uma cena de satélite para iniciar a análise.")

# ============================================================
# PIPELINE STATUS (SEM TERNÁRIOS)
# ============================================================
st.divider()
st.subheader("🚀 Project Pipeline")
has_search = len(st.session_state.search_results) > 0
has_scene = st.session_state.satellite_data is not None
has_change = st.session_state.change_result is not None
has_ai = len(st.session_state.object_detections) > 0

cols = st.columns(5)

with cols[0]:
    if has_search: st.success("✅ Search")
    else: st.info("⏳ Search")

with cols[1]:
    if has_scene: st.success("✅ Download")
    else: st.info("⏳ Download")

with cols[2]:
    if has_scene: st.success("✅ Spectral")
    else: st.info("⏳ Spectral")

with cols[3]:
    if has_change: st.success("✅ Changes")
    else: st.info("⏳ Changes")

with cols[4]:
    if has_ai: st.success("✅ AI")
    elif has_scene: st.info("🧠 AI Ready")
    else: st.info("⏳ AI")

st.caption("Satellite Geospatial Intelligence • Earth Observation • Computer Vision • Geospatial AI")