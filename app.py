"""
app.py — Satellite Geospatial Intelligence
Orchestration only: page config, state, search/download/analysis wiring.
Presentation lives in ui/. Scientific logic lives in src/.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st

from src.catalog import search_sentinel, create_bbox
from src.config import RAW_DIR
from src.downloader import download_required_bands
from src.geospatial import read_band, align_band_to_reference
from src.raster_validation import validate_raster
from src.visualization import create_rgb, create_false_color
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi
from src.index_visualization import create_index_figure
from src.classification import classify_land_cover, calculate_class_percentages
from src.land_cover import create_land_cover_figure, calculate_area_km2
from src.change_detection import (
    calculate_difference,
    detect_change,
    calculate_change_statistics,
)
from src.change_visualization import create_change_figure
from src.object_detection import (
    normalize_rgb,
    validate_detection_image,
    filter_detections,
    filter_classes,
    draw_detections,
)
from src.tiling import create_tiles
from src.detector_model import SatelliteDetector
from src.geospatial_detections import georeference_detections, to_geojson_bytes
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
from ui.status import init_pipeline_status, update_pipeline_status

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
init_pipeline_status()

# ------------------------------------------------------------
# Session defaults
# ------------------------------------------------------------
_DEFAULTS = {
    "search_results": [],
    "satellite_data": None,
    "drawn_aoi": None,
    "rgb_img": None,
    "false_color_img": None,
    "ndvi": None,
    "ndwi": None,
    "ndbi": None,
    "index_stats": {},
    "index_figure": None,
    "classification_fig": None,
    "percentages": None,
    "area_data": None,
    "detection_rgb": None,
    "object_detections": [],
    "detection_figure": None,
    "change_result": None,
    "transform": None,
    "crs": None,
    "export_geojson": False,
}
for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
render_header()

# ------------------------------------------------------------
# Mission Control (sidebar)
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("**Mission Control**")
    st.caption("AOI · temporal window · scene filter")

    st.markdown("AOI")
    latitude = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0, value=-23.5505, format="%.6f"
    )
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, value=-46.6333, format="%.6f"
    )
    area_size = st.slider("Area size (deg)", 0.01, 0.20, 0.05, 0.01)

    st.markdown("Temporal window")
    start_date = st.date_input("Start", value=date(2026, 1, 1))
    end_date = st.date_input("End", value=date(2026, 8, 23))

    st.markdown("Scene filter")
    max_cloud_cover = st.slider("Max cloud cover", 0, 100, 10, 1, format="%d%%")

    search_clicked = st.button("Search Sentinel-2", type="primary", use_container_width=True)

if search_clicked:
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    drawn_aoi_for_search = st.session_state.get("drawn_aoi")
    with st.spinner("Searching Sentinel-2 catalog..."):
        try:
            results = search_sentinel(
                latitude=latitude,
                longitude=longitude,
                area_size=area_size,
                start_date=str(start_date),
                end_date=str(end_date),
                max_cloud_cover=max_cloud_cover,
                bbox=drawn_aoi_for_search["bbox"] if drawn_aoi_for_search else None,
            )
            st.session_state.search_results = results
            st.session_state.satellite_data = None
            st.session_state.change_result = None
            st.session_state.object_detections = []
            update_pipeline_status("Catalog", "done")
            st.rerun()
        except Exception as e:
            st.error("Satellite catalog search failed.")
            st.exception(e)

items = st.session_state.search_results
drawn_aoi = st.session_state.drawn_aoi

# ------------------------------------------------------------
# Mission summary + map
# ------------------------------------------------------------
render_mission_summary(items, drawn_aoi, latitude, longitude, area_size)
render_geospatial_operations_center(lambda: render_map_panel())

# Capture AOI drawn on the map (if any)
try:
    aoi = get_selected_aoi()
    if aoi is not None:
        st.session_state.drawn_aoi = aoi
except Exception:
    pass


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

        b02 = align_band_to_reference(b02, m02, b04, m04)
        b03 = align_band_to_reference(b03, m03, b04, m04)
        b08 = align_band_to_reference(b08, m08, b04, m04)
        b11 = align_band_to_reference(b11, m11, b04, m04)

        for band, label in [
            (b02, "B02"),
            (b03, "B03"),
            (b04, "B04"),
            (b08, "B08"),
            (b11, "B11"),
        ]:
            validate_raster(band, label=label)

        rgb = create_rgb(blue=b02, green=b03, red=b04)
        false_color = create_false_color(green=b03, red=b04, nir=b08)
        st.session_state.rgb_img = rgb
        st.session_state.false_color_img = false_color

        ndvi = calculate_ndvi(red=b04, nir=b08)
        ndwi = calculate_ndwi(green=b03, nir=b08)
        ndbi = calculate_ndbi(nir=b08, swir=b11)

        ndvi_s = _index_stats(ndvi)
        ndwi_s = _index_stats(ndwi)
        ndbi_s = _index_stats(ndbi)
        st.session_state.ndvi = ndvi_s["mean"] if ndvi_s else None
        st.session_state.ndwi = ndwi_s["mean"] if ndwi_s else None
        st.session_state.ndbi = ndbi_s["mean"] if ndbi_s else None
        st.session_state.index_stats = {"ndvi": ndvi_s, "ndwi": ndwi_s, "ndbi": ndbi_s}

        st.session_state.index_figure = create_index_figure(
            ndvi, "NDVI — Vegetation", cmap="RdYlGn"
        )

        classification = classify_land_cover(ndvi=ndvi, ndwi=ndwi, ndbi=ndbi)
        st.session_state.classification_fig = create_land_cover_figure(classification)
        st.session_state.percentages = calculate_class_percentages(classification)
        st.session_state.area_data = calculate_area_km2(
            classification, pixel_size_meters=10.0
        )

        detection_rgb = normalize_rgb(red=b04, green=b03, blue=b02)
        validate_detection_image(detection_rgb)
        st.session_state.detection_rgb = detection_rgb
        st.session_state.transform = m04["transform"]
        st.session_state.crs = str(m04["crs"])
    except Exception as e:
        st.error("Failed to process satellite bands.")
        st.exception(e)


def download_callback(item) -> None:
    bbox = (
        st.session_state.drawn_aoi["bbox"]
        if st.session_state.drawn_aoi
        else create_bbox(latitude, longitude, area_size)
    )
    with st.spinner(f"Downloading {item.id}..."):
        try:
            bands = download_required_bands(
                item=item,
                bbox=bbox,
                output_directory=RAW_DIR / item.id,
            )
            st.session_state.satellite_data = {
                "scene_id": item.id,
                "date": str(item.datetime.date()) if item.datetime else "Unknown",
                "cloud": float(item.properties.get("eo:cloud_cover", 0)),
                "bands": bands,
                "latitude": latitude,
                "longitude": longitude,
                "area_size": area_size,
            }
            st.session_state.change_result = None
            st.session_state.object_detections = []
            _process_bands()
            update_pipeline_status("Imagery", "done")
            update_pipeline_status("Spectral", "done")
            st.rerun()
        except Exception as e:
            st.error("Download failed.")
            st.exception(e)


# ------------------------------------------------------------
# Archive + active scene
# ------------------------------------------------------------
render_scene_catalog(items, download_callback)
render_active_scene(
    st.session_state.satellite_data,
    st.session_state.rgb_img,
    st.session_state.false_color_img,
)

# ------------------------------------------------------------
# Analysis (tabs — presentation only)
# ------------------------------------------------------------
tab_spectral, tab_land, tab_change, tab_ai = st.tabs(
    ["Spectral", "Land Cover", "Change", "Geospatial AI"]
)

with tab_spectral:
    render_spectral_intelligence(
        st.session_state.ndvi,
        st.session_state.ndwi,
        st.session_state.ndbi,
        st.session_state.index_figure,
        st.session_state.get("index_stats"),
    )

with tab_land:
    render_land_cover(
        st.session_state.classification_fig,
        st.session_state.percentages,
        st.session_state.area_data,
    )


def run_change_detection(params, bbox) -> None:
    before_name = params["before_name"]
    after_name = params["after_name"]
    threshold = params["threshold"]
    index_choice = params["index_choice"]
    scene_options = params.get("scene_options") or {}

    if not scene_options:
        for it in items:
            date_str = str(it.datetime.date()) if it.datetime else "Unknown"
            cloud = float(it.properties.get("eo:cloud_cover", 0))
            label = f"{date_str} · {cloud:.2f}% · {it.id[:12]}"
            scene_options[label] = it

    before_item = scene_options.get(before_name)
    after_item = scene_options.get(after_name)
    if not before_item or not after_item or before_item.id == after_item.id:
        st.warning("Choose two different scenes.")
        return

    try:
        with st.spinner("Downloading Data A..."):
            before_bands = download_required_bands(
                item=before_item,
                bbox=bbox,
                output_directory=RAW_DIR / before_item.id,
            )
        with st.spinner("Downloading Data B..."):
            after_bands = download_required_bands(
                item=after_item,
                bbox=bbox,
                output_directory=RAW_DIR / after_item.id,
            )

        b04_b, m04_b = read_band(before_bands["B04"])
        b03_b, m03_b = read_band(before_bands["B03"])
        b08_b, m08_b = read_band(before_bands["B08"])
        b11_b, m11_b = read_band(before_bands["B11"])
        b04_a, m04_a = read_band(after_bands["B04"])
        b03_a, m03_a = read_band(after_bands["B03"])
        b08_a, m08_a = read_band(after_bands["B08"])
        b11_a, m11_a = read_band(after_bands["B11"])

        b03_b = align_band_to_reference(b03_b, m03_b, b04_b, m04_b)
        b08_b = align_band_to_reference(b08_b, m08_b, b04_b, m04_b)
        b11_b = align_band_to_reference(b11_b, m11_b, b04_b, m04_b)
        b03_a = align_band_to_reference(b03_a, m03_a, b04_a, m04_a)
        b08_a = align_band_to_reference(b08_a, m08_a, b04_a, m04_a)
        b11_a = align_band_to_reference(b11_a, m11_a, b04_a, m04_a)

        if "NDVI" in index_choice:
            before_idx = calculate_ndvi(b04_b, b08_b)
            after_idx = calculate_ndvi(b04_a, b08_a)
        elif "NDWI" in index_choice:
            before_idx = calculate_ndwi(b03_b, b08_b)
            after_idx = calculate_ndwi(b03_a, b08_a)
        else:
            before_idx = calculate_ndbi(b08_b, b11_b)
            after_idx = calculate_ndbi(b08_a, b11_a)

        diff = calculate_difference(
            before_idx, after_idx, before_metadata=m04_b, after_metadata=m04_a
        )
        change_map = detect_change(diff, threshold=threshold)
        stats = calculate_change_statistics(change_map, pixel_size_meters=10.0)
        fig = create_change_figure(change_map, title=f"{index_choice} Change Detection")

        st.session_state.change_result = {
            "statistics": stats,
            "figure": fig,
            "index_name": index_choice,
        }
        update_pipeline_status("Change", "done")
        st.success("Change detection completed.")
    except Exception as e:
        st.error("Change detection failed.")
        st.exception(e)


with tab_change:
    change_params = render_change_detection_controls(items)
    if change_params is not None:
        bbox = (
            st.session_state.drawn_aoi["bbox"]
            if st.session_state.drawn_aoi
            else create_bbox(latitude, longitude, area_size)
        )
        run_change_detection(change_params, bbox)
        st.rerun()
    render_change_detection_results()


def run_ai_inference(params, detection_rgb) -> None:
    if detection_rgb is None:
        st.warning("RGB image for AI is not available.")
        return

    try:
        detector = SatelliteDetector(model_id=params["model_id"], device="cpu")
        tiles = create_tiles(
            detection_rgb,
            tile_size=params["tile_size"],
            overlap=params["overlap"],
        )
        detections = detector.predict_tiles(tiles, confidence=params["confidence"])
        detections = filter_detections(detections, params["confidence"])
        detections = filter_classes(detections, params["classes"])
        st.session_state.object_detections = detections
        if detections:
            st.session_state.detection_figure = draw_detections(
                detection_rgb, detections
            )
        update_pipeline_status("AI", "done")
        st.success(f"AI inference completed. {len(detections)} objects detected.")
    except Exception as e:
        st.error("AI inference failed.")
        st.exception(e)


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

render_pipeline_status()
render_footer()
