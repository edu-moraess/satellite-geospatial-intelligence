"""
app.py — Satellite Geospatial Intelligence

Orchestration only:
page configuration, session state, AOI control,
search/download/analysis wiring.

Presentation lives in ui/.
Scientific logic lives in src/.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st
from streamlit.errors import StreamlitAPIException

# ============================================================
# INTERFACES DE SENSOR
# ============================================================
from src.catalog_interface import search_sensor_catalog
from src.download_interface import download_sensor_bands
from src.sensor_registry import get_sensor, list_sensors, SENSORS

# ============================================================
# MÓDULOS EXISTENTES
# ============================================================
from src.config import RAW_DIR
from src.catalog import create_bbox

from src.geospatial import (
    read_band,
    align_band_to_reference,
)

from src.raster_validation import (
    validate_raster,
    validate_raster_pair,
    RasterValidationError,
)

from src.visualization import (
    create_rgb,
    create_false_color,
)

from src.spectral import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
)

from src.index_visualization import create_index_figure

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
    draw_detections,
)

from src.tiling import create_tiles
from src.detector_model import SatelliteDetector

from src.geospatial_detections import (
    georeference_detections,
    to_geojson_bytes,
)

from src.map_view import render_map_panel
from src.aoi import get_selected_aoi

# ============================================================
# UI — NOVA IDENTIDADE VISUAL
# ============================================================
from ui.theme import load_theme

from ui.layout import (
    section_header,
    metric_card,
    render_pipeline,
    status_badge,
)

from ui.components import (
    render_header,
    render_mission_control,
    render_catalog_table,
    render_spectral_cards,
    render_change_metrics,
    render_ai_config,
)

from ui.status import (
    init_pipeline_status,
    update_pipeline_status,
    get_pipeline_status,
    render_pipeline_status,
    status_indicator,
)

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Carrega o tema CSS
st.markdown(load_theme(), unsafe_allow_html=True)

init_pipeline_status()


# ============================================================
# SESSION DEFAULTS
# ============================================================

_DEFAULTS = {
    # AOI
    "aoi_latitude": -23.5505,
    "aoi_longitude": -46.6333,
    "aoi_area_size": 0.05,
    "aoi_source": "manual",
    "_last_processed_map_click": None,

    # Catalog / imagery
    "search_results": [],
    "satellite_data": None,
    "drawn_aoi": None,

    # Visualization
    "rgb_img": None,
    "false_color_img": None,

    # Spectral
    "ndvi": None,
    "ndwi": None,
    "ndbi": None,
    "index_stats": {},
    "index_figure": None,

    # Land cover
    "classification_fig": None,
    "percentages": None,
    "area_data": None,

    # AI
    "detection_rgb": None,
    "object_detections": [],
    "detection_figure": None,

    # Change detection
    "change_result": None,

    # Georeferencing
    "transform": None,
    "crs": None,

    # Export
    "export_geojson": False,
}

for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize map click coordinates
if "_map_click_lat" not in st.session_state:
    st.session_state["_map_click_lat"] = None
if "_map_click_lon" not in st.session_state:
    st.session_state["_map_click_lon"] = None


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# ANALYSIS CONTROL — SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding-bottom:0.5rem;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:1rem;">
            <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#8FA3AD;">Mission Control</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SELETOR DE SENSOR
    # --------------------------------------------------------
    st.markdown("**Sensor**")
    sensor_options = {sensor.name: sensor.id for sensor in SENSORS.values()}
    selected_sensor_name = st.selectbox("", list(sensor_options.keys()))
    current_sensor_id = sensor_options[selected_sensor_name]
    current_sensor = get_sensor(current_sensor_id)
    st.caption(current_sensor.description)

    st.markdown("---")
    st.markdown("**Area of Interest**")

    # Callbacks to clear map click when manual input changes
    def clear_map_click():
        st.session_state["_map_click_lat"] = None
        st.session_state["_map_click_lon"] = None

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        key="aoi_latitude",
        format="%.6f",
        help="Analysis area center latitude.",
        on_change=clear_map_click,
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        key="aoi_longitude",
        format="%.6f",
        help="Analysis area center longitude.",
        on_change=clear_map_click,
    )

    area_size = st.slider(
        "Area size (deg)",
        min_value=0.01,
        max_value=0.20,
        step=0.01,
        key="aoi_area_size",
        help="Approximate AOI side length in degrees.",
    )

    st.caption(
        "Coordinates can be entered manually or selected "
        "directly on the map."
    )

    st.markdown("---")
    st.markdown("**Temporal Window**")

    start_date = st.date_input(
        "Start",
        value=date(2026, 1, 1),
    )

    end_date = st.date_input(
        "End",
        value=date(2026, 8, 23),
    )

    st.markdown("---")
    st.markdown("**Scene Filter**")

    max_cloud_cover = st.slider(
        "Max cloud cover",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        format="%d%%",
    )

    search_clicked = st.button(
        "🔍 Search Satellite Data",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# AOI VALUES AFTER WIDGET STATE UPDATE
# ============================================================
# Use map click coordinates if available, else widget values
if st.session_state.get("_map_click_lat") is not None:
    latitude = float(st.session_state["_map_click_lat"])
else:
    latitude = float(st.session_state["aoi_latitude"])

if st.session_state.get("_map_click_lon") is not None:
    longitude = float(st.session_state["_map_click_lon"])
else:
    longitude = float(st.session_state["aoi_longitude"])

area_size = float(st.session_state["aoi_area_size"])


# ============================================================
# SATELLITE SEARCH (GENERIC)
# ============================================================

if search_clicked or st.session_state.get("retry_search", False):

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    drawn_aoi_for_search = st.session_state.get("drawn_aoi")

    with st.spinner(f"Searching {current_sensor.name} catalog..."):
        try:
            results = search_sensor_catalog(
                sensor_id=current_sensor_id,
                latitude=latitude,
                longitude=longitude,
                area_size=area_size,
                start_date=str(start_date),
                end_date=str(end_date),
                max_cloud_cover=max_cloud_cover,
                bbox=(
                    drawn_aoi_for_search["bbox"]
                    if drawn_aoi_for_search
                    else None
                ),
                max_retries=3,
                max_items=30,
            )

            st.session_state.search_results = results
            st.session_state.satellite_data = None
            st.session_state.change_result = None
            st.session_state.object_detections = []
            st.session_state.detection_figure = None
            st.session_state["retry_search"] = False

            update_pipeline_status("Catalog", "done")
            st.rerun()

        except Exception as error:
            st.session_state["retry_search"] = True
            update_pipeline_status("Catalog", "error")
            st.error(
                f"Falha na busca ao catálogo {current_sensor.name}. "
                "O serviço pode estar sobrecarregado. "
                "Tente novamente ou ajuste os filtros (ex.: aumentar cobertura de nuvens)."
            )
            if st.button("Tentar novamente", key="retry_button"):
                st.session_state["retry_search"] = True
                st.rerun()
            with st.expander("Detalhes do erro"):
                st.exception(error)


# ============================================================
# CURRENT DATA
# ============================================================

items = st.session_state.search_results
drawn_aoi = st.session_state.drawn_aoi


# ============================================================
# 1. MISSION CONTROL (resumo)
# ============================================================

def render_mission_summary_integrated():
    """Versão integrada do resumo da missão usando os novos componentes."""

    num_scenes = len(items) if items else 0

    # Dados para o resumo
    aoi_data = {
        "lat": f"{latitude:.4f}",
        "lon": f"{longitude:.4f}",
        "area": f"{area_size:.2f}° × {area_size:.2f}°",
        "time_window": f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}",
        "cloud_coverage": f"{max_cloud_cover}% max",
        "scenes": num_scenes,
    }

    render_mission_control(aoi_data)


render_mission_summary_integrated()


# ============================================================
# 2. GEOSPATIAL OPERATIONS (mapa)
# ============================================================

def map_panel_wrapper():
    """
    Wrapper seguro para renderizar o mapa.
    Captura StreamlitAPIException causada pela tentativa
    de modificar estado de widget já instanciado.
    """
    try:
        state = render_map_panel(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
            key="aoi_map",
        )

        aoi = get_selected_aoi(state)

        if aoi:
            st.session_state.drawn_aoi = aoi
        else:
            st.session_state.drawn_aoi = None

        return state

    except StreamlitAPIException:
        st.warning(
            "A interação com o mapa está temporariamente indisponível "
            "devido a um conflito de estado do Streamlit. "
            "Você ainda pode usar as coordenadas manuais."
        )
        return None


# Seção Geospatial Operations
section_header("Geospatial Operations", "Interactive Earth Observation Map")

# Container do mapa
map_placeholder = st.container()
with map_placeholder:
    map_panel_wrapper()

# Informações auxiliares do mapa
map_cols = st.columns(4)
map_info = [
    ("AOI", f"{latitude:.4f}, {longitude:.4f}"),
    ("AREA", f"{area_size:.2f}° × {area_size:.2f}°"),
    ("BBOX", "—"),
    ("ZOOM", "12"),
]
for col, (label, value) in zip(map_cols, map_info):
    with col:
        st.markdown(
            f"""
            <div style="padding:0.2rem 0;">
                <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:#8FA3AD;">{label}</div>
                <div style="font-size:0.85rem;color:#E8EEF2;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# ============================================================
# 3. SATELLITE CATALOG
# ============================================================

section_header("Satellite Catalog", f"{len(items)} observations" if items else "No observations")

if items:
    # Converte items para o formato da tabela
    scenes = []
    for item in items:
        date_str = str(item.datetime.date()) if item.datetime else "Unknown"
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        scenes.append({
            "date": date_str,
            "cloud": f"{cloud:.2f}%",
            "status": "Ready" if cloud < 10 else "Pending",
        })
    render_catalog_table(scenes)
else:
    st.caption("Nenhuma cena encontrada. Ajuste os filtros e tente novamente.")

st.markdown("---")


# ============================================================
# 4. ACTIVE OBSERVATION
# ============================================================

section_header("Active Observation", "Select a scene from the catalog above")

# Se houver dados carregados, exibe as imagens
if st.session_state.satellite_data is not None and st.session_state.rgb_img is not None:
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:0.5rem;">
                <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;color:#8FA3AD;margin-bottom:0.3rem;">Natural Color</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(st.session_state.rgb_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_img2:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:0.5rem;">
                <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;color:#8FA3AD;margin-bottom:0.3rem;">False Color</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(st.session_state.false_color_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.caption("Baixe uma cena do catálogo para visualizar as composições.")

st.markdown("---")


# ============================================================
# INDEX STATISTICS
# ============================================================

def _index_stats(arr: np.ndarray) -> dict | None:

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return None

    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


# ============================================================
# BAND PROCESSING
# ============================================================

def _process_bands() -> None:

    data = st.session_state.satellite_data

    if data is None:
        return

    try:

        b02, m02 = read_band(data["bands"]["B02"])
        b03, m03 = read_band(data["bands"]["B03"])
        b04, m04 = read_band(data["bands"]["B04"])
        b08, m08 = read_band(data["bands"]["B08"])
        b11, m11 = read_band(data["bands"]["B11"])

        # Align all bands to B04 / 10 m grid
        b02 = align_band_to_reference(b02, m02, b04, m04)
        b03 = align_band_to_reference(b03, m03, b04, m04)
        b08 = align_band_to_reference(b08, m08, b04, m04)
        b11 = align_band_to_reference(b11, m11, b04, m04)

        # Validate
        for band, label in [
            (b02, "B02"),
            (b03, "B03"),
            (b04, "B04"),
            (b08, "B08"),
            (b11, "B11"),
        ]:
            validate_raster(band, label=label)

        # RGB
        rgb = create_rgb(blue=b02, green=b03, red=b04)
        false_color = create_false_color(green=b03, red=b04, nir=b08)

        st.session_state.rgb_img = rgb
        st.session_state.false_color_img = false_color

        # Spectral indices
        ndvi = calculate_ndvi(red=b04, nir=b08)
        ndwi = calculate_ndwi(green=b03, nir=b08)
        ndbi = calculate_ndbi(nir=b08, swir=b11)

        ndvi_s = _index_stats(ndvi)
        ndwi_s = _index_stats(ndwi)
        ndbi_s = _index_stats(ndbi)

        st.session_state.ndvi = ndvi_s["mean"] if ndvi_s else None
        st.session_state.ndwi = ndwi_s["mean"] if ndwi_s else None
        st.session_state.ndbi = ndbi_s["mean"] if ndbi_s else None

        st.session_state.index_stats = {
            "ndvi": ndvi_s,
            "ndwi": ndwi_s,
            "ndbi": ndbi_s,
        }

        st.session_state.index_figure = create_index_figure(
            ndvi,
            "NDVI — Vegetation",
            cmap="RdYlGn",
        )

        # Land cover
        classification = classify_land_cover(ndvi=ndvi, ndwi=ndwi, ndbi=ndbi)

        st.session_state.classification_fig = create_land_cover_figure(classification)
        st.session_state.percentages = calculate_class_percentages(classification)
        st.session_state.area_data = calculate_area_km2(
            classification,
            pixel_size_meters=float(current_sensor.resolution),
        )

        # AI RGB
        detection_rgb = normalize_rgb(red=b04, green=b03, blue=b02)
        validate_detection_image(detection_rgb)
        st.session_state.detection_rgb = detection_rgb

        # Georeferencing
        st.session_state.transform = m04["transform"]
        st.session_state.crs = str(m04["crs"])

    except RasterValidationError as error:
        st.error("Raster validation failed.")
        st.warning(str(error))

    except Exception as error:
        st.error("Failed to process satellite bands.")
        st.exception(error)


# ============================================================
# DOWNLOAD CALLBACK
# ============================================================

def download_callback(item) -> None:

    current_latitude = float(st.session_state["aoi_latitude"])
    current_longitude = float(st.session_state["aoi_longitude"])
    current_area_size = float(st.session_state["aoi_area_size"])

    bbox = (
        st.session_state.drawn_aoi["bbox"]
        if st.session_state.drawn_aoi
        else create_bbox(
            current_latitude,
            current_longitude,
            current_area_size,
        )
    )

    with st.spinner(f"Downloading {item.id}..."):
        try:
            bands = download_sensor_bands(
                sensor_id=current_sensor_id,
                item=item,
                bbox=bbox,
                output_directory=RAW_DIR / item.id,
            )

            st.session_state.satellite_data = {
                "scene_id": item.id,
                "date": (
                    str(item.datetime.date())
                    if item.datetime
                    else "Unknown"
                ),
                "cloud": float(item.properties.get("eo:cloud_cover", 0)),
                "bands": bands,
                "latitude": current_latitude,
                "longitude": current_longitude,
                "area_size": current_area_size,
            }

            st.session_state.change_result = None
            st.session_state.object_detections = []
            st.session_state.detection_figure = None

            _process_bands()

            update_pipeline_status("Imagery", "done")
            update_pipeline_status("Spectral", "done")

            st.rerun()

        except RasterValidationError as error:
            st.error("Downloaded imagery failed raster validation.")
            st.warning(str(error))

        except Exception as error:
            st.error("Download failed.")
            st.exception(error)


# ============================================================
# SCENE CATALOG (botões de download integrados)
# ============================================================

def render_scene_catalog_integrated(items, download_callback):
    """Versão integrada do catálogo com botões de download."""
    if not items:
        return

    # Tabela com botões
    for idx, item in enumerate(items):
        date_str = str(item.datetime.date()) if item.datetime else "Unknown"
        cloud = float(item.properties.get("eo:cloud_cover", 0))
        cloud_str = f"{cloud:.2f}%"

        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        with col1:
            st.write(date_str)
        with col2:
            st.write(cloud_str)
        with col3:
            status = "✓ Ready" if cloud < 10 else "⏳ Pending"
            st.write(status)
        with col4:
            if st.button("Download", key=f"download_{idx}"):
                download_callback(item)
                st.rerun()


# Catálogo com botões
section_header("Scene Catalog", f"{len(items)} observations available" if items else "No observations")

if items:
    render_scene_catalog_integrated(items, download_callback)
else:
    st.caption("Nenhuma cena encontrada.")

st.markdown("---")


# ============================================================
# 5. SPECTRAL INTELLIGENCE
# ============================================================

section_header("Spectral Intelligence", "Index analysis")

# Cards dos índices
if st.session_state.ndvi is not None:
    indices = {
        "NDVI": f"{st.session_state.ndvi:.3f}",
        "NDWI": f"{st.session_state.ndwi:.3f}" if st.session_state.ndwi is not None else "—",
        "NDBI": f"{st.session_state.ndbi:.3f}" if st.session_state.ndbi is not None else "—",
    }
    render_spectral_cards(indices)

# Mapa do índice
if st.session_state.index_figure is not None:
    st.plotly_chart(st.session_state.index_figure, use_container_width=True)
else:
    st.caption("Nenhum índice disponível. Baixe uma cena primeiro.")

st.markdown("---")


# ============================================================
# 6. LAND COVER
# ============================================================

section_header("Land Cover", "Classification from spectral indices")

if st.session_state.classification_fig is not None:
    st.plotly_chart(st.session_state.classification_fig, use_container_width=True)

    # Estatísticas de cobertura
    if st.session_state.percentages:
        cols = st.columns(5)
        labels = ["Vegetation", "Water", "Built-up", "Bare Soil", "Other"]
        for col, label in zip(cols, labels):
            val = st.session_state.percentages.get(label, 0)
            with col:
                st.markdown(
                    f"""
                    <div class="panel" style="text-align:center;padding:0.4rem 0.2rem;">
                        <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:#8FA3AD;">{label}</div>
                        <div style="font-size:1.1rem;font-weight:500;color:#E8EEF2;">{val:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    st.caption("Nenhuma classificação disponível. Baixe uma cena primeiro.")

st.markdown("---")


# ============================================================
# 7. CHANGE DETECTION
# ============================================================

def run_change_detection(params, bbox) -> None:

    before_name = params["before_name"]
    after_name = params["after_name"]
    threshold = float(params["threshold"])
    index_choice = params["index_choice"]

    scene_options = params.get("scene_options") or {}

    if not scene_options:
        for item in items:
            date_str = str(item.datetime.date()) if item.datetime else "Unknown"
            cloud = float(item.properties.get("eo:cloud_cover", 0))
            label = f"{date_str} · {cloud:.2f}% · {item.id[:12]}"
            scene_options[label] = item

    before_item = scene_options.get(before_name)
    after_item = scene_options.get(after_name)

    if before_item is None or after_item is None:
        st.warning("The selected scenes could not be resolved from the current catalog.")
        return

    if before_item.id == after_item.id:
        st.warning("Choose two different scenes.")
        return

    try:
        with st.spinner("Downloading Before scene..."):
            before_bands = download_sensor_bands(
                sensor_id=current_sensor_id,
                item=before_item,
                bbox=bbox,
                output_directory=RAW_DIR / before_item.id,
            )

        with st.spinner("Downloading After scene..."):
            after_bands = download_sensor_bands(
                sensor_id=current_sensor_id,
                item=after_item,
                bbox=bbox,
                output_directory=RAW_DIR / after_item.id,
            )

        # Read Before
        b04_b, m04_b = read_band(before_bands["B04"])
        b03_b, m03_b = read_band(before_bands["B03"])
        b08_b, m08_b = read_band(before_bands["B08"])
        b11_b, m11_b = read_band(before_bands["B11"])

        # Read After
        b04_a, m04_a = read_band(after_bands["B04"])
        b03_a, m03_a = read_band(after_bands["B03"])
        b08_a, m08_a = read_band(after_bands["B08"])
        b11_a, m11_a = read_band(after_bands["B11"])

        # Validate base rasters
        validate_raster(b04_b, m04_b, label="Before B04")
        validate_raster(b04_a, m04_a, label="After B04")

        # Align bands within each scene
        b03_b = align_band_to_reference(b03_b, m03_b, b04_b, m04_b)
        b08_b = align_band_to_reference(b08_b, m08_b, b04_b, m04_b)
        b11_b = align_band_to_reference(b11_b, m11_b, b04_b, m04_b)

        b03_a = align_band_to_reference(b03_a, m03_a, b04_a, m04_a)
        b08_a = align_band_to_reference(b08_a, m08_a, b04_a, m04_a)
        b11_a = align_band_to_reference(b11_a, m11_a, b04_a, m04_a)

        # Validate intra-scene bands
        for band, label in [
            (b03_b, "Before B03"),
            (b08_b, "Before B08"),
            (b11_b, "Before B11"),
            (b03_a, "After B03"),
            (b08_a, "After B08"),
            (b11_a, "After B11"),
        ]:
            validate_raster(band, label=label)

        # Calculate indices
        if "NDVI" in index_choice:
            before_idx = calculate_ndvi(red=b04_b, nir=b08_b)
            after_idx = calculate_ndvi(red=b04_a, nir=b08_a)
            index_name = "NDVI"
        elif "NDWI" in index_choice:
            before_idx = calculate_ndwi(green=b03_b, nir=b08_b)
            after_idx = calculate_ndwi(green=b03_a, nir=b08_a)
            index_name = "NDWI"
        else:
            before_idx = calculate_ndbi(nir=b08_b, swir=b11_b)
            after_idx = calculate_ndbi(nir=b08_a, swir=b11_a)
            index_name = "NDBI"

        validate_raster(before_idx, label=f"Before {index_name}")
        validate_raster(after_idx, label=f"After {index_name}")

        # Align After index to Before grid
        with st.spinner("Aligning Before / After spatial grids..."):
            after_idx_aligned = align_band_to_reference(
                band_array=after_idx,
                band_metadata=m04_a,
                reference_array=before_idx,
                reference_metadata=m04_b,
            )

        # Build metadata
        before_idx_metadata = m04_b.copy()
        before_idx_metadata.update({
            "height": before_idx.shape[0],
            "width": before_idx.shape[1],
            "transform": m04_b["transform"],
            "crs": m04_b["crs"],
            "nodata": np.nan,
            "dtype": str(before_idx.dtype),
        })

        after_idx_metadata = m04_b.copy()
        after_idx_metadata.update({
            "height": after_idx_aligned.shape[0],
            "width": after_idx_aligned.shape[1],
            "transform": m04_b["transform"],
            "crs": m04_b["crs"],
            "nodata": np.nan,
            "dtype": str(after_idx_aligned.dtype),
        })

        # Final pair validation
        validation = validate_raster_pair(
            before_idx,
            after_idx_aligned,
            metadata_a=before_idx_metadata,
            metadata_b=after_idx_metadata,
            label_a=f"Before {index_name}",
            label_b=f"After {index_name}",
            require_same_dtype=False,
        )

        # Difference: AFTER - BEFORE
        diff = calculate_difference(
            before=before_idx,
            after=after_idx_aligned,
            before_metadata=before_idx_metadata,
            after_metadata=after_idx_metadata,
        )

        validate_raster(diff, label=f"{index_name} difference")

        change_map = detect_change(diff, threshold=threshold)
        stats = calculate_change_statistics(
            change_map,
            pixel_size_meters=float(current_sensor.resolution),
        )

        fig = create_change_figure(change_map, title=f"{index_choice} Change Detection")

        st.session_state.change_result = {
            "statistics": stats,
            "figure": fig,
            "index_name": index_choice,
            "before_scene_id": before_item.id,
            "after_scene_id": after_item.id,
            "before_date": str(before_item.datetime.date()) if before_item.datetime else "Unknown",
            "after_date": str(after_item.datetime.date()) if after_item.datetime else "Unknown",
            "threshold": threshold,
            "validation": validation,
            "alignment": {
                "reference": "Before",
                "target": "After",
                "shape": before_idx.shape,
                "overlap_fraction": validation["overlap_fraction"],
            },
        }

        update_pipeline_status("Change", "done")
        st.success("Change detection completed successfully.")

    except RasterValidationError as error:
        st.session_state.change_result = None
        update_pipeline_status("Change", "error")
        st.error("Change detection stopped by raster validation.")
        st.warning(str(error))

    except Exception as error:
        st.session_state.change_result = None
        update_pipeline_status("Change", "error")
        st.error("Change detection failed.")
        st.exception(error)


section_header("Change Intelligence", "Before / After comparison")

# Controles
col_before, col_after, col_index, col_threshold = st.columns(4)

with col_before:
    before_label = st.selectbox(
        "Before",
        options=[f"{item.datetime.date()} · {item.properties.get('eo:cloud_cover', 0):.2f}%" for item in items] if items else ["—"],
        key="before_select",
    )
with col_after:
    after_label = st.selectbox(
        "After",
        options=[f"{item.datetime.date()} · {item.properties.get('eo:cloud_cover', 0):.2f}%" for item in items] if items else ["—"],
        key="after_select",
    )
with col_index:
    index_choice = st.selectbox(
        "Index",
        options=["NDVI", "NDWI", "NDBI"],
        key="change_index",
    )
with col_threshold:
    threshold = st.slider("Threshold", 0.05, 0.30, 0.10, 0.01, key="change_threshold")

if st.button("📊 Analyze Change", use_container_width=False):
    if items and before_label != "—" and after_label != "—":
        bbox = (
            st.session_state.drawn_aoi["bbox"]
            if st.session_state.drawn_aoi
            else create_bbox(
                float(st.session_state["aoi_latitude"]),
                float(st.session_state["aoi_longitude"]),
                float(st.session_state["aoi_area_size"]),
            )
        )

        scene_options = {}
        for item in items:
            date_str = str(item.datetime.date()) if item.datetime else "Unknown"
            cloud = float(item.properties.get("eo:cloud_cover", 0))
            label = f"{date_str} · {cloud:.2f}%"
            scene_options[label] = item

        params = {
            "before_name": before_label,
            "after_name": after_label,
            "index_choice": index_choice,
            "threshold": threshold,
            "scene_options": scene_options,
        }

        run_change_detection(params, bbox)
        st.rerun()

# Resultados da mudança
if st.session_state.change_result:
    stats = st.session_state.change_result["statistics"]
    render_change_metrics(
        f"{stats.get('decrease_area_km2', 0):.3f} km²",
        f"{stats.get('increase_area_km2', 0):.3f} km²",
        f"{stats.get('total_changed_km2', 0):.3f} km²",
    )

    if st.session_state.change_result.get("figure"):
        st.plotly_chart(st.session_state.change_result["figure"], use_container_width=True)
else:
    st.caption("Nenhuma análise de mudança realizada. Selecione duas cenas e clique em 'Analyze Change'.")

st.markdown("---")


# ============================================================
# 8. GEOSPATIAL AI
# ============================================================

def run_ai_inference(params, detection_rgb) -> None:

    if detection_rgb is None:
        st.warning("RGB image for AI is not available.")
        return

    try:

        detector = SatelliteDetector(
            model_id=params["model_id"],
            device="cpu",
        )

        tiles = create_tiles(
            detection_rgb,
            tile_size=params["tile_size"],
            overlap=params["overlap"],
        )

        detections = detector.predict_tiles(
            tiles,
            confidence=params["confidence"],
        )

        detections = filter_detections(detections, params["confidence"])
        detections = filter_classes(detections, params["classes"])

        st.session_state.object_detections = detections

        if detections:
            st.session_state.detection_figure = draw_detections(detection_rgb, detections)
        else:
            st.session_state.detection_figure = None

        update_pipeline_status("AI", "done")
        st.success(f"AI inference completed. {len(detections)} objects detected.")

    except Exception as error:
        update_pipeline_status("AI", "error")
        st.error("AI inference failed.")
        st.exception(error)


section_header("Geospatial AI", "Object detection & inference")

# Configuração
with st.expander("Model Configuration", expanded=False):
    model_options = ["remote_sensing_detector", "yolo_satellite", "custom_resnet"]
    model_id = st.selectbox("Model", model_options, key="ai_model")
    confidence = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05, key="ai_confidence")
    tile_size = st.selectbox("Tile Size", [256, 512, 1024], index=1, key="ai_tile")
    overlap = st.slider("Overlap", 0.0, 0.5, 0.2, 0.05, key="ai_overlap")
    classes = st.multiselect(
        "Classes",
        ["Vegetation", "Water", "Built-up", "Bare Soil", "Other"],
        default=["Vegetation", "Built-up"],
        key="ai_classes",
    )

if st.button("🧠 Run Geospatial AI", use_container_width=False):
    if st.session_state.detection_rgb is not None:
        params = {
            "model_id": model_id,
            "confidence": confidence,
            "tile_size": tile_size,
            "overlap": overlap,
            "classes": classes,
        }
        run_ai_inference(params, st.session_state.detection_rgb)
        st.rerun()
    else:
        st.warning("Nenhuma imagem disponível para IA. Baixe uma cena primeiro.")

# Resultados da IA
if st.session_state.detection_figure is not None:
    st.image(st.session_state.detection_figure, use_container_width=True)

    # Botão de exportação
    if st.session_state.object_detections:
        st.session_state["export_geojson"] = True

if st.session_state.get("export_geojson", False):
    detections = st.session_state.object_detections

    if detections and st.session_state.transform is not None:
        gdf = georeference_detections(
            detections,
            transform=st.session_state.transform,
            crs=st.session_state.crs,
        )
        geojson_bytes = to_geojson_bytes(gdf)

        st.download_button(
            "📥 Download GeoJSON",
            data=geojson_bytes,
            file_name="detections.geojson",
            mime="application/geo+json",
        )

    st.session_state["export_geojson"] = False

st.markdown("---")


# ============================================================
# 9. PROCESSING PIPELINE
# ============================================================

section_header("Processing Pipeline", "Current stage")

# Obtém status atualizado
pipeline_status = get_pipeline_status()
render_pipeline_status(pipeline_status)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="margin-top:2rem;padding-top:0.8rem;border-top:1px solid rgba(255,255,255,0.08);font-size:0.65rem;color:#8FA3AD;text-align:center;letter-spacing:0.04em;">
        Satellite Geospatial Intelligence · Earth Observation · Remote Sensing · Geospatial Analytics
        <br>
        Spectral values are analytical measurements; interpret with sensor, resolution and preprocessing context.
    </div>
    """,
    unsafe_allow_html=True,
)