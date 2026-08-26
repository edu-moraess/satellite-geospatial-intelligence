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

# ============================================================
# INTERFACES DE SENSOR (NOVAS)
# ============================================================
from src.catalog_interface import search_sensor_catalog
from src.download_interface import download_sensor_bands
from src.sensor_registry import get_sensor, list_sensors

# ============================================================
# MÓDULOS EXISTENTES
# ============================================================
from src.config import RAW_DIR
from src.aoi import create_bbox

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

from ui.theme import apply_theme

from ui.layout import (
    render_header,
    render_mission_summary,
    render_geospatial_operations_center,
    render_scene_catalog,
    render_active_scene,
    render_spectral_intelligence,
    render_land_cover,
    render_change_detection_controls,
    render_change_detection_results,
    render_geospatial_ai_controls,
    render_geospatial_ai_results,
    render_pipeline_status,
    render_footer,
)

from ui.status import (
    init_pipeline_status,
    update_pipeline_status,
)

# ============================================================
# SENSOR CONFIGURATION
# ============================================================

# Por enquanto, usamos apenas Sentinel-2.
CURRENT_SENSOR_ID = "sentinel2"

_current_sensor = get_sensor(CURRENT_SENSOR_ID)
if _current_sensor is None:
    raise RuntimeError(f"Sensor '{CURRENT_SENSOR_ID}' não encontrado no registro.")

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
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


# ============================================================
# AOI HELPERS
# ============================================================

def _current_aoi_center() -> tuple[float, float]:
    return (
        float(st.session_state["aoi_latitude"]),
        float(st.session_state["aoi_longitude"]),
    )


def _current_area_size() -> float:
    return float(st.session_state["aoi_area_size"])


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# ANALYSIS CONTROL — SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("**Analysis Control**")
    st.caption("AOI · temporal window · scene filter")

    # Exibe o sensor atual
    st.markdown(f"**Sensor:** {_current_sensor.name}")
    st.caption(_current_sensor.description)

    st.markdown("AOI")

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        key="aoi_latitude",
        format="%.6f",
        help="Analysis area center latitude.",
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        key="aoi_longitude",
        format="%.6f",
        help="Analysis area center longitude.",
    )

    area_size = st.slider(
        "Area size (deg)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        key="aoi_area_size",
        help="Approximate AOI side length in degrees.",
    )

    st.caption(
        "Coordinates can be entered manually or selected "
        "directly on the map."
    )

    st.markdown("Temporal window")

    start_date = st.date_input(
        "Start",
        value=date(2026, 1, 1),
    )

    end_date = st.date_input(
        "End",
        value=date(2026, 8, 23),
    )

    st.markdown("Scene filter")

    max_cloud_cover = st.slider(
        "Max cloud cover",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        format="%d%%",
    )

    search_clicked = st.button(
        "Search Sentinel-2",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# AOI VALUES AFTER WIDGET STATE UPDATE
# ============================================================

latitude = float(st.session_state["aoi_latitude"])
longitude = float(st.session_state["aoi_longitude"])
area_size = float(st.session_state["aoi_area_size"])


# ============================================================
# SENTINEL-2 SEARCH
# ============================================================

if search_clicked:

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    drawn_aoi_for_search = st.session_state.get("drawn_aoi")

    with st.spinner("Searching Sentinel-2 catalog..."):
        try:
            results = search_sensor_catalog(
                sensor_id=CURRENT_SENSOR_ID,
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
            )

            st.session_state.search_results = results
            st.session_state.satellite_data = None
            st.session_state.change_result = None
            st.session_state.object_detections = []
            st.session_state.detection_figure = None

            update_pipeline_status("Catalog", "done")
            st.rerun()

        except Exception as error:
            st.error("Satellite catalog search failed.")
            st.exception(error)


# ============================================================
# CURRENT DATA
# ============================================================

items = st.session_state.search_results
drawn_aoi = st.session_state.drawn_aoi


# ============================================================
# MISSION SUMMARY + MAP
# ============================================================

render_mission_summary(
    items,
    drawn_aoi,
    latitude,
    longitude,
    area_size,
)


def map_panel_wrapper():

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


render_geospatial_operations_center(map_panel_wrapper)


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

        # ----------------------------------------------------
        # ALIGN ALL BANDS TO B04 / 10 m GRID
        # ----------------------------------------------------

        b02 = align_band_to_reference(b02, m02, b04, m04)
        b03 = align_band_to_reference(b03, m03, b04, m04)
        b08 = align_band_to_reference(b08, m08, b04, m04)
        b11 = align_band_to_reference(b11, m11, b04, m04)

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        for band, label in [
            (b02, "B02"),
            (b03, "B03"),
            (b04, "B04"),
            (b08, "B08"),
            (b11, "B11"),
        ]:
            validate_raster(band, label=label)

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = create_rgb(blue=b02, green=b03, red=b04)
        false_color = create_false_color(green=b03, red=b04, nir=b08)

        st.session_state.rgb_img = rgb
        st.session_state.false_color_img = false_color

        # ----------------------------------------------------
        # SPECTRAL INDICES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LAND COVER
        # ----------------------------------------------------

        classification = classify_land_cover(ndvi=ndvi, ndwi=ndwi, ndbi=ndbi)

        st.session_state.classification_fig = create_land_cover_figure(classification)
        st.session_state.percentages = calculate_class_percentages(classification)
        st.session_state.area_data = calculate_area_km2(classification, pixel_size_meters=10.0)

        # ----------------------------------------------------
        # AI RGB
        # ----------------------------------------------------

        detection_rgb = normalize_rgb(red=b04, green=b03, blue=b02)
        validate_detection_image(detection_rgb)
        st.session_state.detection_rgb = detection_rgb

        # ----------------------------------------------------
        # GEOREFERENCING
        # ----------------------------------------------------

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
                sensor_id=CURRENT_SENSOR_ID,
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
# SCENE CATALOG + ACTIVE SCENE
# ============================================================

render_scene_catalog(items, download_callback)

render_active_scene(
    st.session_state.satellite_data,
    st.session_state.rgb_img,
    st.session_state.false_color_img,
)


# ============================================================
# ANALYSIS TABS
# ============================================================

(
    tab_spectral,
    tab_land,
    tab_change,
    tab_ai,
) = st.tabs(
    [
        "Spectral",
        "Land Cover",
        "Change",
        "Geospatial AI",
    ]
)


# ============================================================
# SPECTRAL
# ============================================================

with tab_spectral:

    render_spectral_intelligence(
        st.session_state.ndvi,
        st.session_state.ndwi,
        st.session_state.ndbi,
        st.session_state.index_figure,
        st.session_state.get("index_stats"),
    )


# ============================================================
# LAND COVER
# ============================================================

with tab_land:

    render_land_cover(
        st.session_state.classification_fig,
        st.session_state.percentages,
        st.session_state.area_data,
    )


# ============================================================
# CHANGE DETECTION
# ============================================================

def run_change_detection(params, bbox) -> None:
    """Execute robust Before / After change detection."""

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
                sensor_id=CURRENT_SENSOR_ID,
                item=before_item,
                bbox=bbox,
                output_directory=RAW_DIR / before_item.id,
            )

        with st.spinner("Downloading After scene..."):
            after_bands = download_sensor_bands(
                sensor_id=CURRENT_SENSOR_ID,
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
        stats = calculate_change_statistics(change_map, pixel_size_meters=10.0)

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


with tab_change:

    change_params = render_change_detection_controls(items)

    if change_params is not None:

        bbox = (
            st.session_state.drawn_aoi["bbox"]
            if st.session_state.drawn_aoi
            else create_bbox(
                float(st.session_state["aoi_latitude"]),
                float(st.session_state["aoi_longitude"]),
                float(st.session_state["aoi_area_size"]),
            )
        )

        run_change_detection(change_params, bbox)
        st.rerun()

    render_change_detection_results()


# ============================================================
# GEOSPATIAL AI
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


with tab_ai:

    ai_params = render_geospatial_ai_controls(
        st.session_state.satellite_data,
        st.session_state.detection_rgb,
    )

    if ai_params is not None:
        run_ai_inference(ai_params, st.session_state.detection_rgb)
        st.rerun()

    render_geospatial_ai_results(
        st.session_state.object_detections,
        st.session_state.detection_rgb,
    )


# ============================================================
# GEOJSON EXPORT
# ============================================================

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
            "Download GeoJSON",
            data=geojson_bytes,
            file_name="detections.geojson",
            mime="application/geo+json",
        )

    st.session_state["export_geojson"] = False


# ============================================================
# FOOTER
# ============================================================

render_pipeline_status()
render_footer()