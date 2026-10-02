"""GEOCORE — Geospatial Computing & Intelligence Platform application shell.

This module owns orchestration only. Scientific operations live in src/;
reusable presentation lives in ui/.
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from streamlit.errors import StreamlitAPIException

from src.aoi import get_selected_aoi
from src.catalog import create_bbox
from src.catalog_interface import search_sensor_catalog
from src.config import RAW_DIR
from src.download_interface import download_sensor_bands
from src.geospatial_detections import georeference_detections, to_geojson_bytes
from src.model_registry import list_models, get_model, model_available
from src.object_detection import filter_classes, filter_detections, draw_detections
from src.tiling import create_tiles
from src.detector_model import SatelliteDetector
from src.raster_validation import RasterValidationError
from src.workflows.imagery import process_scene
from src.workflows.change import run_change
from src.map_view import render_map_panel
from src.deep_learning.inference import build_inference_gate
from src.deep_learning.prithvi_burn_scars import runtime_available

from ui.catalog import render_scene_catalog
from ui.components import render_header, render_spectral_cards, render_change_metrics
from ui.layout import section_header
from ui.mission_control import render_sidebar
from ui.status import init_pipeline_status, update_pipeline_status, get_pipeline_status, render_pipeline_status
from ui.theme import load_theme


st.set_page_config(page_title="GEOCORE", page_icon="G", layout="wide", initial_sidebar_state="expanded")
st.markdown(load_theme(), unsafe_allow_html=True)
init_pipeline_status()

DEFAULTS = {
    "search_results": [], "drawn_aoi": None, "active_scene_id": None,
    "satellite_data": None, "rgb_img": None, "false_color_img": None,
    "ndvi": None, "ndwi": None, "ndbi": None, "index_stats": {},
    "index_figure": None, "classification_fig": None, "percentages": None,
    "area_data": None, "detection_rgb": None, "object_detections": [],
    "detection_figure": None, "transform": None, "crs": None,
    "change_result": None, "retry_search": False, "scene_quality": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

config = render_sidebar()
sensor = config["sensor"]

# Map clicks can update the live AOI without changing the scientific pipeline.
latitude = float(st.session_state.get("_map_click_lat") or config["latitude"])
longitude = float(st.session_state.get("_map_click_lon") or config["longitude"])
area_size = config["area_size"]


def reset_analysis_state() -> None:
    for key, value in {
        "satellite_data": None, "active_scene_id": None, "rgb_img": None,
        "false_color_img": None, "ndvi": None, "ndwi": None, "ndbi": None,
        "index_stats": {}, "index_figure": None, "classification_fig": None,
        "percentages": None, "area_data": None, "detection_rgb": None,
        "object_detections": [], "detection_figure": None, "transform": None,
        "crs": None, "change_result": None, "scene_quality": None,
    }.items():
        st.session_state[key] = value
    st.session_state["pipeline_status"] = {k: "pending" for k in ["Catalog", "Imagery", "Spectral", "Change", "AI"]}


def perform_search() -> None:
    start_date = config["start_date"]
    end_date = config["end_date"]

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    drawn_aoi = st.session_state.get("drawn_aoi") or {}
    bbox = drawn_aoi.get("bbox")
    if bbox is None:
        bbox = create_bbox(latitude, longitude, area_size)

    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        st.error("Invalid AOI. Check latitude, longitude and area size.")
        return

    # A new query is authoritative: do not keep stale scenes from a previous AOI.
    st.session_state.search_results = []
    st.session_state.active_scene_id = None
    st.session_state.retry_search = False
    update_pipeline_status("Catalog", "active")

    with st.spinner(f"Searching {sensor.name} catalog..."):
        try:
            results = search_sensor_catalog(
                sensor_id=config["sensor_id"],
                latitude=latitude,
                longitude=longitude,
                area_size=area_size,
                start_date=str(start_date),
                end_date=str(end_date),
                max_cloud_cover=config["max_cloud_cover"],
                bbox=bbox,
                max_retries=3,
                max_items=30,
            )

            st.session_state.search_results = results or []
            reset_analysis_state()
            update_pipeline_status("Catalog", "done")

            if not st.session_state.search_results:
                st.warning(
                    f"No {sensor.name} scenes found for this AOI and filters. "
                    f"Try a larger area, wider date range or higher cloud-cover limit."
                )
                return

            st.success(f"{len(st.session_state.search_results)} scene(s) found in the satellite catalog.")
            st.rerun()

        except Exception as exc:
            update_pipeline_status("Catalog", "error")
            st.error(
                f"Could not access the {sensor.name} satellite catalog. "
                "The query was not completed."
            )
            with st.expander("Technical details"):
                st.exception(exc)


if config["search_clicked"]:
    perform_search()

items = st.session_state.search_results

render_header()
section_header("Geospatial Workspace", "AOI · Scenes · Observations")
try:
    map_state = render_map_panel(latitude=latitude, longitude=longitude, area_size=area_size, key="aoi_map")
    selected_aoi = get_selected_aoi(map_state)
    if selected_aoi:
        st.session_state.drawn_aoi = selected_aoi
except StreamlitAPIException:
    st.warning("Map interaction is temporarily unavailable. Manual coordinates remain available.")

map_cols = st.columns(3)
for col, label, value in zip(map_cols, ["AOI", "AREA", "SCENES"], [f"{latitude:.4f}, {longitude:.4f}", f"{area_size:.2f}° × {area_size:.2f}°", str(len(items))]):
    with col:
        st.markdown(f'<div class="telemetry"><div class="metric-label">{label}</div><div class="metric-value-small">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Archive is a single source of truth. Selection and download happen here only.
selected = render_scene_catalog(items, st.session_state.active_scene_id)
if selected is not None:
    st.session_state.active_scene_id = selected.id

active_item = next((x for x in items if x.id == st.session_state.active_scene_id), None)
if active_item is not None:
    section_header("Observation", f"{active_item.datetime.date() if active_item.datetime else 'Unknown'} · {float(active_item.properties.get('eo:cloud_cover', 0)):.2f}%")
    meta_cols = st.columns([3, 2, 1.2])
    with meta_cols[0]: st.caption(active_item.id)
    with meta_cols[1]: st.caption(sensor.name)
    with meta_cols[2]:
        if st.button("Download Scene", type="primary", use_container_width=True, key="download_active"):
            bbox = st.session_state.drawn_aoi["bbox"] if st.session_state.drawn_aoi else create_bbox(latitude, longitude, area_size)
            with st.spinner(f"Downloading {active_item.id}..."):
                try:
                    bands = download_sensor_bands(sensor_id=config["sensor_id"], item=active_item, bbox=bbox, output_directory=RAW_DIR / active_item.id)
                    data = {"scene_id": active_item.id, "date": str(active_item.datetime.date()) if active_item.datetime else "Unknown", "cloud": float(active_item.properties.get("eo:cloud_cover", 0)), "bands": bands, "latitude": latitude, "longitude": longitude, "area_size": area_size}
                    processed = process_scene(data, sensor.resolution)
                    st.session_state.satellite_data = data
                    for key, value in processed.items():
                        st.session_state[key] = value
                    st.session_state.object_detections = []
                    st.session_state.detection_figure = None
                    update_pipeline_status("Imagery", "done")
                    update_pipeline_status("Spectral", "done")
                    st.rerun()
                except RasterValidationError as exc:
                    update_pipeline_status("Imagery", "error"); st.error("Downloaded imagery failed raster validation."); st.warning(str(exc))
                except Exception as exc:
                    update_pipeline_status("Imagery", "error"); st.error("Scene download or processing failed.")
                    with st.expander("Technical details"): st.exception(exc)
else:
    section_header("Observation", "No scene selected")
    st.caption("Select a scene from the archive to download and analyze it.")

if st.session_state.rgb_img is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.image(st.session_state.rgb_img, caption="Natural Color", use_container_width=True)
    with c2:
        st.image(st.session_state.false_color_img, caption="False Color · NIR", use_container_width=True)

quality = st.session_state.get("scene_quality")
if quality:
    section_header("Scene Quality", "SCL · cloud · usable pixels")
    if quality.get("available"):
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Usable", f'{100.0 * (quality.get("valid_fraction") or 0.0):.1f}%')
        with q2:
            st.metric("Cloud", f'{100.0 * (quality.get("cloud_fraction") or 0.0):.1f}%')
        with q3:
            st.metric("Shadow", f'{100.0 * (quality.get("shadow_fraction") or 0.0):.1f}%')
        with q4:
            st.metric("Quality", f'{100.0 * (quality.get("score") or 0.0):.1f}%')
        st.caption(
            "Quality is AOI-level and derived from Sentinel-2 SCL. "
            "It is not a probability of scene correctness."
        )
    else:
        st.caption("SCL quality layer is unavailable for this observation.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

section_header("Spectral", "NDVI · NDWI · NDBI")
if st.session_state.index_stats:
    stats = st.session_state.index_stats
    render_spectral_cards({
        "NDVI": f'{stats.get("ndvi", {}).get("mean", 0.0):.3f}',
        "NDWI": f'{stats.get("ndwi", {}).get("mean", 0.0):.3f}',
        "NDBI": f'{stats.get("ndbi", {}).get("mean", 0.0):.3f}',
    })
    if st.session_state.index_figure is not None:
        st.plotly_chart(st.session_state.index_figure, use_container_width=True)
else:
    st.caption("No spectral analysis available. Download a scene first.")

section_header("Land Cover", "Spectral classification")
if st.session_state.classification_fig is not None:
    st.plotly_chart(st.session_state.classification_fig, use_container_width=True)
    if st.session_state.percentages:
        cols = st.columns(5)
        for col, label in zip(cols, ["Vegetation", "Water", "Built-up", "Bare Soil", "Other"]):
            with col: st.metric(label, f'{st.session_state.percentages.get(label, 0):.1f}%')
else:
    st.caption("No land-cover classification available. Download a scene first.")

section_header("Change", "Before / After")
if len(items) >= 2:
    labels = {f'{i.id[:14]} · {i.datetime.date() if i.datetime else "Unknown"} · {float(i.properties.get("eo:cloud_cover",0)):.2f}%': i for i in items}
    c1,c2,c3,c4 = st.columns(4)
    with c1: before_label = st.selectbox("Before", list(labels), key="change_before")
    with c2: after_label = st.selectbox("After", list(labels), index=1, key="change_after")
    with c3: index_name = st.selectbox("Index", ["NDVI","NDWI","NDBI"], key="change_index")
    with c4: threshold = st.slider("Threshold", 0.05, 0.30, 0.10, 0.01, key="change_threshold")
    if st.button("Analyze Change", key="analyze_change", type="primary"):
        before_item, after_item = labels[before_label], labels[after_label]
        if before_item.id == after_item.id:
            st.warning("Choose two different scenes.")
        else:
            bbox = st.session_state.drawn_aoi["bbox"] if st.session_state.drawn_aoi else create_bbox(latitude, longitude, area_size)
            try:
                with st.spinner("Downloading and aligning Before / After scenes..."):
                    before_bands = download_sensor_bands(sensor_id=config["sensor_id"], item=before_item, bbox=bbox, output_directory=RAW_DIR / before_item.id)
                    after_bands = download_sensor_bands(sensor_id=config["sensor_id"], item=after_item, bbox=bbox, output_directory=RAW_DIR / after_item.id)
                    result = run_change(before_bands, after_bands, index_name, threshold, sensor.resolution)
                result.update({"index_name": index_name, "before_scene_id": before_item.id, "after_scene_id": after_item.id, "before_date": str(before_item.datetime.date()) if before_item.datetime else "Unknown", "after_date": str(after_item.datetime.date()) if after_item.datetime else "Unknown", "threshold": threshold})
                st.session_state.change_result = result
                update_pipeline_status("Change", "done")
            except Exception as exc:
                update_pipeline_status("Change", "error"); st.error("Change detection failed.")
                with st.expander("Technical details"): st.exception(exc)
else:
    st.caption("At least two scenes are required for change detection.")
if st.session_state.change_result:
    stats = st.session_state.change_result["statistics"]
    render_change_metrics(f'{stats.get("decrease_area_km2",0):.3f} km²', f'{stats.get("increase_area_km2",0):.3f} km²', f'{stats.get("total_changed_km2",0):.3f} km²')
    st.plotly_chart(st.session_state.change_result["figure"], use_container_width=True)

section_header("Geospatial AI", "Real checkpoint required")
models = list_models()
model_id = st.selectbox("Model", models, key="ai_model")
model = get_model(model_id)
st.caption(model.description)
if not model_available(model_id):
    st.info("MODEL UNAVAILABLE — no checkpoint is registered for this model. No synthetic detections will be generated.")
else:
    c1,c2,c3 = st.columns(3)
    with c1: confidence = st.slider("Confidence",0.1,0.9,0.5,0.05,key="ai_confidence")
    with c2: tile_size = st.selectbox("Tile size",[256,512,1024],index=1,key="ai_tile")
    with c3: overlap = st.slider("Overlap",0.0,0.5,0.2,0.05,key="ai_overlap")
    classes = st.multiselect("Classes", list(model.classes), default=list(model.classes), key="ai_classes")
    if st.button("Run Geospatial AI", type="primary", key="run_ai"):
        try:
            tiles = create_tiles(st.session_state.detection_rgb, tile_size=tile_size, overlap=overlap)
            detections = SatelliteDetector(model_id=model_id, device="cpu").predict_tiles(tiles, confidence=confidence)
            detections = filter_classes(filter_detections(detections, confidence), classes)
            st.session_state.object_detections = detections
            st.session_state.detection_figure = draw_detections(st.session_state.detection_rgb, detections) if detections else None
            update_pipeline_status("AI", "done")
        except Exception as exc:
            update_pipeline_status("AI", "error"); st.error("Geospatial AI inference failed.")
            with st.expander("Technical details"): st.exception(exc)
if st.session_state.detection_figure is not None:
    st.image(st.session_state.detection_figure, caption="Detected objects", use_container_width=True)
if st.session_state.object_detections and st.session_state.transform is not None:
    gdf = georeference_detections(st.session_state.object_detections, transform=st.session_state.transform, crs=st.session_state.crs)
    st.download_button("Download GeoJSON", data=to_geojson_bytes(gdf), file_name="detections.geojson", mime="application/geo+json", use_container_width=True)

section_header("Deep Learning", "Multispectral foundation-model gate")
deep_model_id = "prithvi_eo_v2_300m_burn_scars"
available_deep_bands = (
    set(st.session_state.satellite_data.get("bands", {}).keys())
    if st.session_state.satellite_data
    else set()
)
deep_runtime_ok, deep_runtime_message = runtime_available()
deep_harmonization_status = st.session_state.get("deep_harmonization_status")

deep_gate = build_inference_gate(
    deep_model_id,
    available_deep_bands,
    source_domain="sentinel2-l2a",
    harmonization_status=deep_harmonization_status,
)

d1, d2, d3 = st.columns(3)
with d1:
    st.metric("Model", "Prithvi-EO 2.0")
with d2:
    st.metric("Input", "6 × 512²")
with d3:
    st.metric("Runtime", "READY" if deep_runtime_ok else "OPTIONAL")

if deep_runtime_ok:
    st.success("Real Prithvi runtime detected.")
else:
    st.info(deep_runtime_message)

if deep_gate.ready:
    st.success("Deep-learning input contract and checkpoint are ready.")
else:
    st.info(deep_gate.reason)
    if deep_gate.compatibility.missing_bands:
        st.caption("Missing bands: " + ", ".join(deep_gate.compatibility.missing_bands))
    else:
        st.caption("Input contract detected: B02 · B03 · B04 · B8A · B11 · B12.")

st.caption(
    "Official checkpoint: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars. "
    "The checkpoint is approximately 1.3 GB and is loaded lazily. "
    "Sentinel-2 inference remains blocked until full HLS S30 harmonization "
    "is validated; spatial resampling alone is insufficient."
)

section_header("Pipeline", "Mission state")
render_pipeline_status(get_pipeline_status())

st.markdown('<div class="footer-note">GEOCORE · Geospatial Computing & Intelligence<br>Analytical measurements require appropriate sensor, resolution and preprocessing context.</div>', unsafe_allow_html=True)
