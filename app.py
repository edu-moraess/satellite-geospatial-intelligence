from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st

from src.catalog import search_sentinel, create_bbox
from src.config import RAW_DIR
from src.downloader import download_required_bands
from src.geospatial import read_band, align_band_to_reference

from src.visualization import (
    create_rgb,
    create_false_color,
)

from src.spectral import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
)

from src.index_visualization import (
    create_index_figure,
)

from src.classification import (
    classify_land_cover,
    calculate_class_percentages,
)

from src.land_cover import (
    create_land_cover_figure,
    calculate_area_km2,
)

from src.change_detection import (
    calculate_difference,
    detect_change,
    calculate_change_statistics,
)

from src.change_visualization import (
    create_change_figure,
)

from src.object_detection import (
    normalize_rgb,
    validate_detection_image,
    filter_detections,
    filter_classes,
    detection_summary,
    draw_detections,
)

from src.tiling import (
    create_tiles,
    tile_count,
)

from src.detector_model import (
    SatelliteDetector,
)

from src.model_registry import (
    list_models,
    get_model,
)

from src.map_view import (
    render_map_panel,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS DARK GEOSPATIAL (Ajustado para Plotly e Pydeck)
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0a0f16;
        color: #e0e0e0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00d4ff !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }

    h1, h2, h3 {
        color: #ffffff !important;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.05);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00d4ff;
    }

    [data-testid="stMetricLabel"] {
        color: #9ca3af;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.5rem;
    }

    .stButton > button {
        background-color: #1f2937;
        color: #e0e0e0;
        border: 1px solid #374151;
        border-radius: 6px;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        border-color: #00d4ff;
        color: #00d4ff;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        color: #ffffff;
        border: none;
        font-weight: bold;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 15px rgba(0, 180, 216, 0.5);
    }

    div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input {
        background-color: #1f2937;
        border-color: #374151;
        color: white;
    }

    div[data-baseweb="slider"] > div {
        color: #00d4ff;
    }

    .streamlit-expanderHeader {
        color: #9ca3af;
        background-color: #111827;
        border-radius: 6px;
        border: 1px solid #1f2937;
    }

    .stImage img, .stPlotlyChart, .stPydeckChart {
        border: 1px solid #1f2937;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }

    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
        background-color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "search_results": [], 
    "satellite_data": None,
    "change_result": None,
    "object_detections": [],
    "map_state": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# HELPER: ALIGN ARRAYS FOR CHANGE DETECTION
# ============================================================

def align_arrays_for_change(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before = np.asarray(before)
    after = np.asarray(after)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("Change detection requires 2D arrays.")
    h = min(before.shape[0], after.shape[0])
    w = min(before.shape[1], after.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("Invalid array dimensions after alignment.")
    return before[:h, :w], after[:h, :w]

# ============================================================
# HEADER
# ============================================================
st.title("🛰️ Satellite Geospatial Intelligence")
st.caption("Earth Observation • Computer Vision • Geospatial AI")

# ============================================================
# SIDEBAR – SEARCH PARAMETERS
# ============================================================
st.sidebar.header("📍 Area of Interest")

latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, -23.5505, format="%.6f")
longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, -46.6333, format="%.6f")
area_size = st.sidebar.slider("Area size (degrees)", 0.01, 0.20, 0.05, 0.01)

st.sidebar.header("📅 Satellite Date Range")
start_date = st.sidebar.date_input("Start date", date(2026, 1, 1))
end_date = st.sidebar.date_input("End date", date(2026, 8, 23))

st.sidebar.header("☁️ Image Quality")
max_cloud_cover = st.sidebar.slider("Maximum cloud coverage", 0, 100, 10, 1, format="%d%%")

# ============================================================
# SEARCH BUTTON
# ============================================================
if st.sidebar.button("🔎 Search Satellite Data", type="primary", use_container_width=True):
    if start_date > end_date:
        st.error("❌ Start date must be before the end date.")
        st.stop()

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
            with st.expander("Technical details"):
                st.exception(error)
            st.stop()

items = st.session_state.search_results
data = st.session_state.satellite_data

# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================
if items:
    st.success(f"🛰️ {len(items)} satellite scenes found.")
    st.subheader("Available Sentinel‑2 Scenes")

    for index, item in enumerate(items[:20]):
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        acquisition_date = item.datetime.date() if item.datetime else "Unknown"

        with st.expander(f"{acquisition_date} • {cloud:.2f}% clouds"):
            st.write(f"**Scene ID:** `{item.id}`")
            st.write(f"**Acquisition:** `{acquisition_date}`")
            st.write(f"**Cloud coverage:** `{cloud:.2f}%`")

            if st.button("⬇️ Download & Analyze", key=f"download_{index}", use_container_width=True):
                bbox = create_bbox(latitude, longitude, area_size)
                output_dir = RAW_DIR / item.id

                with st.spinner("⬇️ Downloading satellite data..."):
                    try:
                        downloaded = download_required_bands(item, bbox, output_dir)
                    except Exception as error:
                        st.error("❌ Satellite download failed.")
                        with st.expander("Technical details"):
                            st.exception(error)
                        st.stop()

                st.session_state.satellite_data = {
                    "scene_id": item.id,
                    "date": str(acquisition_date),
                    "cloud": cloud,
                    "bands": downloaded,
                    "latitude": latitude,
                    "longitude": longitude,
                    "area_size": area_size,
                }
                st.session_state.change_result = None
                st.session_state.object_detections = []
                st.success("✅ Satellite scene downloaded.")
                st.rerun()

data = st.session_state.satellite_data
detection_rgb = None

# ============================================================
# SELECTED SCENE – ANALYSIS
# ============================================================
if data:
    st.divider()
    st.header("🛰️ Selected Satellite Scene")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Acquisition", data["date"])
    with col2:
        st.metric("Cloud Coverage", f"{data['cloud']:.2f}%")
    with col3:
        st.metric("Scene", data["scene_id"][:24])

    st.divider()
    try:
        scene_bbox = create_bbox(data.get("latitude", latitude), data.get("longitude", longitude), data.get("area_size", area_size))
        render_map_panel(
            latitude=data.get("latitude", latitude),
            longitude=data.get("longitude", longitude),
            area_size=data.get("area_size", area_size),
            bbox=scene_bbox,
            scene_id=data["scene_id"],
            acquisition_date=data["date"],
            cloud_cover=data["cloud"],
            key="main_geospatial_map",
        )
    except Exception as error:
        st.warning("⚠️ Interactive geospatial map could not be rendered.")
        with st.expander("Technical details"):
            st.exception(error)

    with st.spinner("📡 Loading spectral bands..."):
        try:
            b02, m02 = read_band(data["bands"]["B02"])
            b03, m03 = read_band(data["bands"]["B03"])
            b04, m04 = read_band(data["bands"]["B04"])
            b08, m08 = read_band(data["bands"]["B08"])
            b11, m11 = read_band(data["bands"]["B11"])
        except Exception as error:
            st.error("❌ Failed to load satellite bands.")
            with st.expander("Technical details"):
                st.exception(error)
            st.stop()

    with st.spinner("🔄 Aligning spectral grids..."):
        try:
            b02 = align_band_to_reference(b02, m02, b04, m04)
            b03 = align_band_to_reference(b03, m03, b04, m04)
            b08 = align_band_to_reference(b08, m08, b04, m04)
            b11 = align_band_to_reference(b11, m11, b04, m04)
        except Exception as error:
            st.error("❌ Failed to align spectral bands.")
            with st.expander("Technical details"):
                st.exception(error)
            st.stop()

    try:
        rgb = create_rgb(blue=b02, green=b03, red=b04)
        detection_rgb = normalize_rgb(red=b04, green=b03, blue=b02)
        validate_detection_image(detection_rgb)
        false_color = create_false_color(green=b03, red=b04, nir=b08)
    except Exception as error:
        st.error("❌ Failed to create images.")
        with st.expander("Technical details"):
            st.exception(error)
        st.stop()

    st.divider()
    st.header("🌍 Satellite Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 Natural RGB")
        st.image(rgb, use_container_width=True)
    with col2:
        st.subheader("🌱 False Color")
        st.image(false_color, use_container_width=True)

    st.divider()
    st.header("🔬 Multispectral Analysis")
    try:
        ndvi = calculate_ndvi(red=b04, nir=b08)
        ndwi = calculate_ndwi(green=b03, nir=b08)
        ndbi = calculate_ndbi(nir=b08, swir=b11)
    except Exception as error:
        st.error("❌ Failed to calculate spectral indices.")
        with st.expander("Technical details"):
            st.exception(error)
        st.stop()

    valid_ndvi = ndvi[np.isfinite(ndvi)]
    valid_ndwi = ndwi[np.isfinite(ndwi)]
    valid_ndbi = ndbi[np.isfinite(ndbi)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌱 Mean NDVI", f"{np.mean(valid_ndvi):.3f}" if valid_ndvi.size else "N/A")
    with col2:
        st.metric("💧 Mean NDWI", f"{np.mean(valid_ndwi):.3f}" if valid_ndwi.size else "N/A")
    with col3:
        st.metric("🏙️ Mean NDBI", f"{np.mean(valid_ndbi):.3f}" if valid_ndbi.size else "N/A")

    st.divider()
    st.header("🗺️ Land Cover Classification")
    st.caption("Rule‑based multispectral baseline using NDVI, NDWI and NDBI.")
    try:
        classification = classify_land_cover(ndvi=ndvi, ndwi=ndwi, ndbi=ndbi)
        land_cover_figure = create_land_cover_figure(classification)
        st.plotly_chart(land_cover_figure, use_container_width=True)
    except Exception as error:
        st.error("❌ Land‑cover classification failed.")
        with st.expander("Technical details"):
            st.exception(error)
        classification = None

    if classification is not None:
        try:
            percentages = calculate_class_percentages(classification)
            st.subheader("📊 Land Cover Distribution")
            cols = st.columns(5)
            with cols[0]: st.metric("🌳 Vegetation", f"{percentages['Vegetation']:.1f}%")
            with cols[1]: st.metric("💧 Water", f"{percentages['Water']:.1f}%")
            with cols[2]: st.metric("🏙️ Built‑up", f"{percentages['Built-up']:.1f}%")
            with cols[3]: st.metric("🟫 Bare Soil", f"{percentages['Bare Soil']:.1f}%")
            with cols[4]: st.metric("⬜ Other", f"{percentages['Other']:.1f}%")
        except Exception:
            pass

        try:
            area = calculate_area_km2(classification, pixel_size_meters=10.0)
            st.subheader("📐 Estimated Area")
            cols = st.columns(4)
            with cols[0]: st.metric("🌳 Vegetation", f"{area['Vegetation']:.3f} km²")
            with cols[1]: st.metric("💧 Water", f"{area['Water']:.3f} km²")
            with cols[2]: st.metric("🏙️ Built‑up", f"{area['Built-up']:.3f} km²")
            with cols[3]: st.metric("🟫 Bare Soil", f"{area['Bare Soil']:.3f} km²")
        except Exception:
            pass

    st.divider()
    st.header("🔬 Spectral Index Maps")
    selected_index = st.selectbox("Choose index", ["NDVI — Vegetation", "NDWI — Water", "NDBI — Built‑up"], key="main_index")
    if selected_index.startswith("NDVI"):
        index_data, index_title, index_cmap = ndvi, "NDVI — Vegetation", "RdYlGn"
    elif selected_index.startswith("NDWI"):
        index_data, index_title, index_cmap = ndwi, "NDWI — Water", "Blues"
    else:
        index_data, index_title, index_cmap = ndbi, "NDBI — Built‑up", "Oranges"

    try:
        fig = create_index_figure(index_data, index_title, cmap=index_cmap)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("⚠️ Could not render spectral map.")

# ============================================================
# CHANGE DETECTION
# ============================================================
st.divider()
st.header("🛰️ Change Detection")
st.caption("Compare two Sentinel‑2 observations of the same area to identify spectral changes over time.")

if len(items) >= 2:
    scene_map = {}
    for item in items:
        scene_date = item.datetime.date() if item.datetime else "Unknown"
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        label = f"{scene_date} • {cloud:.2f}% clouds • {item.id[:12]}…"
        scene_map[label] = item

    labels = list(scene_map.keys())

    col1, col2 = st.columns(2)
    with col1:
        before_label = st.selectbox("📅 Data A — Before", labels, key="change_before")
    with col2:
        after_label = st.selectbox("📅 Data B — After", labels, index=min(1, len(labels) - 1), key="change_after")

    threshold = st.slider("🎚️ Change sensitivity", 0.01, 0.50, 0.10, 0.01, key="change_threshold")
    change_index_choice = st.selectbox("🔬 Index to compare", ["NDVI — Vegetation", "NDWI — Water", "NDBI — Built‑up"], key="change_index")

    if st.button("🔍 Analyze Changes", type="primary", use_container_width=True):
        before_item = scene_map[before_label]
        after_item = scene_map[after_label]

        if before_item.id == after_item.id:
            st.warning("⚠️ Please choose two different scenes.")
        else:
            bbox = create_bbox(latitude, longitude, area_size)
            try:
                with st.spinner("🛰️ Downloading Data A..."):
                    before_bands = download_required_bands(before_item, bbox, RAW_DIR / before_item.id)
                with st.spinner("🛰️ Downloading Data B..."):
                    after_bands = download_required_bands(after_item, bbox, RAW_DIR / after_item.id)

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
                    "difference": difference,
                    "change_map": change_map,
                    "statistics": statistics,
                    "index_name": index_name,
                    "before_id": before_item.id,
                    "after_id": after_item.id,
                }
                st.success("✅ Change detection completed.")
            except Exception as error:
                st.error("❌ Change detection could not be completed.")
                with st.expander("Technical details"):
                    st.exception(error)
else:
    st.info("ℹ️ Search for at least two satellite scenes to activate Change Detection.")

# ============================================================
# CHANGE RESULTS
# ============================================================
change_result = st.session_state.change_result
if change_result:
    st.subheader(f"📊 {change_result['index_name']}")
    stats = change_result["statistics"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🔴 Decrease", f"{stats['decrease_km2']:.3f} km²")
    with c2: st.metric("🟢 Increase", f"{stats['increase_km2']:.3f} km²")
    with c3: st.metric("🛰️ Total Changed", f"{stats['total_changed_km2']:.3f} km²")

    try:
        fig = create_change_figure(change_result["change_map"], title=f"{change_result['index_name']} Change Detection")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ============================================================
# GEOSPATIAL AI
# ============================================================
st.divider()
st.header("🎯 Geospatial AI")
st.caption("Remote‑sensing computer vision pipeline for object detection.")

if data is None:
    st.info("ℹ️ Download a satellite scene to activate Geospatial AI.")
elif detection_rgb is None:
    st.warning("⚠️ RGB image is unavailable for AI.")
else:
    st.subheader("🛰️ Detection Input")
    st.image(detection_rgb, caption="Sentinel‑2 RGB prepared for Geospatial AI", use_container_width=True)

    st.subheader("🧩 AI Image Tiling")
    col1, col2 = st.columns(2)
    with col1: tile_size = st.selectbox("Tile size", [256, 512, 768, 1024], index=1, key="tile_size")
    with col2: tile_overlap = st.slider("Tile overlap", 0, 256, 64, 16, key="tile_overlap")

    if tile_overlap >= tile_size:
        st.error("❌ Overlap must be smaller than tile size.")
    else:
        try:
            n_tiles = tile_count(detection_rgb, tile_size=tile_size, overlap=tile_overlap)
            st.metric("🧩 Image tiles", n_tiles)
        except Exception as error:
            st.error("❌ Failed to calculate tiles.")
            with st.expander("Technical details"):
                st.exception(error)

    st.subheader("⚙️ Detection Configuration")
    confidence_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05, key="object_confidence")

    st.subheader("🧠 Geospatial AI Model")
    try:
        model_ids = list_models()
        selected_model_id = st.selectbox("Model", model_ids, key="selected_model")
        detector = SatelliteDetector(model_id=selected_model_id, device="cpu")
        model_info = detector.info()

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Model", model_info["model"])
        with c2: st.metric("Input", f"{model_info['input_size']}×{model_info['input_size']}")
        with c3:
            st.success("CHECKPOINT FOUND") if model_info["checkpoint_available"] else st.warning("CHECKPOINT MISSING")

        st.caption(model_info["description"])
    except Exception as error:
        detector = None
        model_info = None
        st.error("❌ Model registry failed.")
        with st.expander("Technical details"):
            st.exception(error)

    if detector is not None:
        detection_classes = st.multiselect("Classes of interest", list(model_info["classes"]), default=list(model_info["classes"][:2]), key="object_classes")

        if st.button("🤖 Run Geospatial AI", type="primary", use_container_width=True):
            if not detection_classes:
                st.warning("⚠️ Select at least one class.")
            elif tile_overlap >= tile_size:
                st.error("⚠️ Invalid tile configuration.")
            elif not model_info["checkpoint_available"]:
                st.info("🧠 Model pipeline is ready, but the selected checkpoint is not installed yet.")
            else:
                try:
                    with st.spinner("🧠 Running Geospatial AI..."):
                        tiles = create_tiles(detection_rgb, tile_size=tile_size, overlap=tile_overlap)
                        detections = detector.predict_tiles(tiles, confidence=confidence_threshold)
                        detections = filter_detections(detections, confidence_threshold)
                        detections = filter_classes(detections, detection_classes)
                        st.session_state.object_detections = detections
                    st.success(f"✅ {len(tiles)} tiles processed.")
                except Exception as error:
                    st.error("❌ AI inference failed.")
                    with st.expander("Technical details"):
                        st.exception(error)

        detections = st.session_state.object_detections
        if detections:
            try:
                summary = detection_summary(detections)
                st.subheader("📊 Detection Results")
                c1, c2 = st.columns(2)
                with c1: st.metric("Objects", len(detections))
                with c2: st.metric("Classes", len(summary))
                
                fig = draw_detections(detection_rgb, detections)
                st.pyplot(fig, use_container_width=True)  # Matplotlib é mais adequado para BBox estáticas

                st.subheader("🏷️ Detected Classes")
                for label, count in summary.items():
                    st.write(f"**{label}:** {count}")
            except Exception as error:
                st.error("❌ Could not render detections.")
                with st.expander("Technical details"):
                    st.exception(error)
        else:
            st.info("🔜 No detections available. Connect a trained checkpoint to activate inference.")

# ============================================================
# PIPELINE STATUS (CORRIGIDO)
# ============================================================
st.divider()
st.subheader("🚀 Project Pipeline")

has_search = len(st.session_state.search_results) > 0
has_scene = st.session_state.satellite_data is not None
has_change = st.session_state.change_result is not None
has_ai = len(st.session_state.object_detections) > 0

cols = st.columns(5)
with cols[0]:
    if has_search: st.success("✅ Scene Search")
    else: st.info("⏳ Scene Search")
with cols[1]:
    if has_scene: st.success("✅ Scene Download")
    else: st.info("⏳ Scene Download")
with cols[2]:
    if has_scene: st.success("✅ Spectral Analysis")
    else: st.info("⏳ Spectral Analysis")
with cols[3]:
    if has_change: st.success("✅ Change Detection")
    else: st.info("⏳ Change Detection")
with cols[4]:
    if has_ai: st.success("✅ AI Inference")
    elif has_scene: st.info("🧠 AI Ready")
    else: st.info("⏳ AI Waiting")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Satellite Geospatial Intelligence • Earth Observation • Computer Vision • Geospatial AI")
st.caption("Spectral values are analytical measurements and should be interpreted according to sensor characteristics, spatial resolution and preprocessing.")