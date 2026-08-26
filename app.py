"""
app.py — Satellite Geospatial Intelligence

Application orchestration layer.

Responsibilities:
- Streamlit page configuration
- Session state
- AOI state
- Sentinel-2 catalog search
- Scene download
- Scientific pipeline orchestration
- Change detection orchestration
- Geospatial AI orchestration
- GeoJSON export

Presentation lives in ui/.
Scientific logic lives in src/.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st

from src.catalog import (
    search_sentinel,
    create_bbox,
)

from src.config import RAW_DIR

from src.downloader import (
    download_required_bands,
)

from src.geospatial import (
    read_band,
    align_band_to_reference,
)

from src.raster_validation import (
    RasterValidationError,
    validate_raster,
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
    draw_detections,
)

from src.tiling import (
    create_tiles,
)

from src.detector_model import (
    SatelliteDetector,
)

from src.geospatial_detections import (
    georeference_detections,
    to_geojson_bytes,
)

from src.map_view import (
    render_map_panel,
)

from src.aoi import (
    get_selected_aoi,
)

from ui.theme import (
    apply_theme,
)

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
# PAGE CONFIGURATION
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
# SESSION STATE
# ============================================================

_DEFAULTS = {
    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    "aoi_latitude": -23.5505,
    "aoi_longitude": -46.6333,
    "aoi_area_size": 0.05,
    "aoi_source": "manual",
    "_last_processed_map_click": None,

    # --------------------------------------------------------
    # CATALOG
    # --------------------------------------------------------

    "search_results": [],
    "search_params": None,
    "drawn_aoi": None,

    # --------------------------------------------------------
    # ACTIVE SCENE
    # --------------------------------------------------------

    "satellite_data": None,

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    "rgb_img": None,
    "false_color_img": None,

    # --------------------------------------------------------
    # SPECTRAL
    # --------------------------------------------------------

    "ndvi": None,
    "ndwi": None,
    "ndbi": None,

    "index_stats": {},
    "index_figure": None,

    # --------------------------------------------------------
    # LAND COVER
    # --------------------------------------------------------

    "classification_fig": None,
    "percentages": None,
    "area_data": None,

    # --------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------

    "change_result": None,

    # --------------------------------------------------------
    # GEOSPATIAL AI
    # --------------------------------------------------------

    "detection_rgb": None,
    "object_detections": [],
    "detection_figure": None,

    # --------------------------------------------------------
    # GEOREFERENCING
    # --------------------------------------------------------

    "transform": None,
    "crs": None,

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    "export_geojson": False,
}


for key, value in _DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# SESSION RESET HELPERS
# ============================================================

def _reset_scene_analysis() -> None:
    """
    Reset all analysis products associated with the active
    scene.

    This prevents stale imagery / indices / AI detections
    from remaining visible after another scene is downloaded.
    """

    st.session_state.rgb_img = None
    st.session_state.false_color_img = None

    st.session_state.ndvi = None
    st.session_state.ndwi = None
    st.session_state.ndbi = None

    st.session_state.index_stats = {}
    st.session_state.index_figure = None

    st.session_state.classification_fig = None
    st.session_state.percentages = None
    st.session_state.area_data = None

    st.session_state.detection_rgb = None
    st.session_state.object_detections = []
    st.session_state.detection_figure = None

    st.session_state.transform = None
    st.session_state.crs = None

    st.session_state.change_result = None
    st.session_state.export_geojson = False


def _reset_pipeline_after_new_search() -> None:
    """
    Reset scene-dependent state after a new catalog search.
    """

    st.session_state.satellite_data = None

    _reset_scene_analysis()

    update_pipeline_status(
        "Imagery",
        "pending",
    )

    update_pipeline_status(
        "Spectral",
        "pending",
    )

    update_pipeline_status(
        "Change",
        "pending",
    )

    update_pipeline_status(
        "AI",
        "pending",
    )


# ============================================================
# AOI HELPERS
# ============================================================

def _current_aoi_center() -> tuple[float, float]:

    return (
        float(
            st.session_state["aoi_latitude"]
        ),
        float(
            st.session_state["aoi_longitude"]
        ),
    )


def _current_area_size() -> float:

    return float(
        st.session_state["aoi_area_size"]
    )


def _current_bbox():
    """
    Return the active AOI bbox.

    Priority:
        1. Polygon / AOI selected on map
        2. Manual latitude / longitude / area size
    """

    drawn_aoi = st.session_state.get(
        "drawn_aoi"
    )

    if drawn_aoi:

        bbox = drawn_aoi.get(
            "bbox"
        )

        if bbox is not None:

            return bbox

    latitude, longitude = (
        _current_aoi_center()
    )

    area_size = (
        _current_area_size()
    )

    return create_bbox(
        latitude,
        longitude,
        area_size,
    )


# ============================================================
# INDEX STATISTICS
# ============================================================

def _index_stats(
    arr: np.ndarray,
) -> dict | None:

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    finite = arr[
        np.isfinite(arr)
    ]

    if finite.size == 0:

        return None

    return {
        "mean": float(
            np.mean(finite)
        ),
        "median": float(
            np.median(finite)
        ),
        "min": float(
            np.min(finite)
        ),
        "max": float(
            np.max(finite)
        ),
        "std": float(
            np.std(finite)
        ),
    }


# ============================================================
# PROCESS ACTIVE SCENE
# ============================================================

def _process_bands() -> None:
    """
    Read, align and process the bands of the active scene.

    Reference grid:
        B04 / 10 m

    B02, B03, B08 and B11 are aligned to B04 before
    downstream pixel-wise operations.
    """

    data = (
        st.session_state.get(
            "satellite_data"
        )
    )

    if data is None:

        return

    try:

        # ----------------------------------------------------
        # READ
        # ----------------------------------------------------

        b02, m02 = read_band(
            data["bands"]["B02"]
        )

        b03, m03 = read_band(
            data["bands"]["B03"]
        )

        b04, m04 = read_band(
            data["bands"]["B04"]
        )

        b08, m08 = read_band(
            data["bands"]["B08"]
        )

        b11, m11 = read_band(
            data["bands"]["B11"]
        )

        # ----------------------------------------------------
        # VALIDATE REFERENCE
        # ----------------------------------------------------

        validate_raster(
            b04,
            m04,
            label="B04 reference",
        )

        # ----------------------------------------------------
        # ALIGN ALL BANDS TO B04
        # ----------------------------------------------------

        b02 = align_band_to_reference(
            b02,
            m02,
            b04,
            m04,
        )

        b03 = align_band_to_reference(
            b03,
            m03,
            b04,
            m04,
        )

        b08 = align_band_to_reference(
            b08,
            m08,
            b04,
            m04,
        )

        b11 = align_band_to_reference(
            b11,
            m11,
            b04,
            m04,
        )

        # ----------------------------------------------------
        # VALIDATE ALIGNED BANDS
        # ----------------------------------------------------

        for band, label in (
            (b02, "B02"),
            (b03, "B03"),
            (b04, "B04"),
            (b08, "B08"),
            (b11, "B11"),
        ):

            validate_raster(
                band,
                label=label,
            )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = create_rgb(
            blue=b02,
            green=b03,
            red=b04,
        )

        false_color = create_false_color(
            green=b03,
            red=b04,
            nir=b08,
        )

        st.session_state.rgb_img = rgb
        st.session_state.false_color_img = (
            false_color
        )

        # ----------------------------------------------------
        # SPECTRAL INDICES
        # ----------------------------------------------------

        ndvi = calculate_ndvi(
            red=b04,
            nir=b08,
        )

        ndwi = calculate_ndwi(
            green=b03,
            nir=b08,
        )

        ndbi = calculate_ndbi(
            nir=b08,
            swir=b11,
        )

        # ----------------------------------------------------
        # INDEX STATISTICS
        # ----------------------------------------------------

        ndvi_stats = _index_stats(
            ndvi
        )

        ndwi_stats = _index_stats(
            ndwi
        )

        ndbi_stats = _index_stats(
            ndbi
        )

        st.session_state.ndvi = (
            ndvi_stats["mean"]
            if ndvi_stats
            else None
        )

        st.session_state.ndwi = (
            ndwi_stats["mean"]
            if ndwi_stats
            else None
        )

        st.session_state.ndbi = (
            ndbi_stats["mean"]
            if ndbi_stats
            else None
        )

        st.session_state.index_stats = {
            "ndvi": ndvi_stats,
            "ndwi": ndwi_stats,
            "ndbi": ndbi_stats,
        }

        # ----------------------------------------------------
        # INDEX FIGURE
        # ----------------------------------------------------

        st.session_state.index_figure = (
            create_index_figure(
                ndvi,
                "NDVI — Vegetation",
                cmap="RdYlGn",
            )
        )

        # ----------------------------------------------------
        # LAND COVER
        # ----------------------------------------------------

        classification = (
            classify_land_cover(
                ndvi=ndvi,
                ndwi=ndwi,
                ndbi=ndbi,
            )
        )

        st.session_state.classification_fig = (
            create_land_cover_figure(
                classification
            )
        )

        st.session_state.percentages = (
            calculate_class_percentages(
                classification
            )
        )

        st.session_state.area_data = (
            calculate_area_km2(
                classification,
                pixel_size_meters=10.0,
            )
        )

        # ----------------------------------------------------
        # AI RGB
        # ----------------------------------------------------

        detection_rgb = normalize_rgb(
            red=b04,
            green=b03,
            blue=b02,
        )

        validate_detection_image(
            detection_rgb
        )

        st.session_state.detection_rgb = (
            detection_rgb
        )

        # ----------------------------------------------------
        # GEOREFERENCING
        # ----------------------------------------------------

        st.session_state.transform = (
            m04["transform"]
        )

        st.session_state.crs = str(
            m04["crs"]
        )

    except RasterValidationError:

        raise

    except Exception:

        raise


# ============================================================
# DOWNLOAD SCENE
# ============================================================

def _download_scene(
    item,
    *,
    label: str = "scene",
):
    """
    Download and process a Sentinel-2 scene.
    """

    current_latitude = float(
        st.session_state[
            "aoi_latitude"
        ]
    )

    current_longitude = float(
        st.session_state[
            "aoi_longitude"
        ]
    )

    current_area_size = float(
        st.session_state[
            "aoi_area_size"
        ]
    )

    bbox = _current_bbox()

    with st.spinner(
        f"Downloading {label} · {item.id}..."
    ):

        bands = download_required_bands(
            item=item,
            bbox=bbox,
            output_directory=(
                RAW_DIR / item.id
            ),
        )

    return {
        "scene_id": item.id,
        "date": (
            str(item.datetime.date())
            if item.datetime
            else "Unknown"
        ),
        "cloud": float(
            item.properties.get(
                "eo:cloud_cover",
                0,
            )
        ),
        "bands": bands,
        "latitude": current_latitude,
        "longitude": current_longitude,
        "area_size": current_area_size,
        "bbox": bbox,
    }


# ============================================================
# DOWNLOAD CALLBACK
# ============================================================

def download_callback(item) -> None:
    """
    Scene archive download callback.
    """

    try:

        # Clear old analysis first so stale products cannot
        # remain associated with the new scene.
        _reset_scene_analysis()

        scene_data = _download_scene(
            item,
            label="active scene",
        )

        st.session_state.satellite_data = (
            scene_data
        )

        _process_bands()

        update_pipeline_status(
            "Imagery",
            "done",
        )

        update_pipeline_status(
            "Spectral",
            "done",
        )

        st.success(
            f"Scene {item.id} loaded successfully."
        )

        st.rerun()

    except RasterValidationError as error:

        st.error(
            "Satellite raster validation failed."
        )

        st.warning(
            str(error)
        )

    except Exception as error:

        st.error(
            "Download or scene processing failed."
        )

        st.exception(error)


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# SIDEBAR — ANALYSIS CONTROL
# ============================================================

with st.sidebar:

    st.markdown(
        "**Analysis Control**"
    )

    st.caption(
        "AOI · temporal window · scene filter"
    )

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    st.markdown("AOI")

    st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        key="aoi_latitude",
        format="%.6f",
        help="Analysis area center latitude.",
    )

    st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        key="aoi_longitude",
        format="%.6f",
        help="Analysis area center longitude.",
    )

    st.slider(
        "Area size (deg)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        key="aoi_area_size",
        help="Approximate AOI side length in degrees.",
    )

    st.caption(
        "Coordinates can be entered manually "
        "or selected directly on the map."
    )

    # --------------------------------------------------------
    # TEMPORAL WINDOW
    # --------------------------------------------------------

    st.markdown(
        "Temporal window"
    )

    start_date = st.date_input(
        "Start",
        value=date(
            2026,
            1,
            1,
        ),
    )

    end_date = st.date_input(
        "End",
        value=date(
            2026,
            8,
            23,
        ),
    )

    # --------------------------------------------------------
    # CLOUD FILTER
    # --------------------------------------------------------

    st.markdown(
        "Scene filter"
    )

    max_cloud_cover = st.slider(
        "Max cloud cover",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        format="%d%%",
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_clicked = st.button(
        "Search Sentinel-2",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# CURRENT AOI VALUES
# ============================================================

latitude = float(
    st.session_state[
        "aoi_latitude"
    ]
)

longitude = float(
    st.session_state[
        "aoi_longitude"
    ]
)

area_size = float(
    st.session_state[
        "aoi_area_size"
    ]
)


# ============================================================
# SENTINEL-2 SEARCH
# ============================================================

if search_clicked:

    if start_date > end_date:

        st.error(
            "Start date must be before end date."
        )

        st.stop()

    drawn_aoi_for_search = (
        st.session_state.get(
            "drawn_aoi"
        )
    )

    search_bbox = None

    if drawn_aoi_for_search:

        search_bbox = (
            drawn_aoi_for_search.get(
                "bbox"
            )
        )

    with st.spinner(
        "Searching Sentinel-2 catalog..."
    ):

        try:

            results = search_sentinel(
                latitude=latitude,
                longitude=longitude,
                area_size=area_size,
                start_date=str(
                    start_date
                ),
                end_date=str(
                    end_date
                ),
                max_cloud_cover=(
                    max_cloud_cover
                ),
                bbox=search_bbox,
            )

            st.session_state.search_results = (
                results
            )

            st.session_state.search_params = {
                "latitude": latitude,
                "longitude": longitude,
                "area_size": area_size,
                "start_date": str(
                    start_date
                ),
                "end_date": str(
                    end_date
                ),
                "max_cloud_cover": (
                    max_cloud_cover
                ),
                "bbox": search_bbox,
            }

            _reset_pipeline_after_new_search()

            update_pipeline_status(
                "Catalog",
                "done",
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Satellite catalog search failed."
            )

            st.exception(error)


# ============================================================
# CURRENT DATA
# ============================================================

items = (
    st.session_state.search_results
)

drawn_aoi = (
    st.session_state.drawn_aoi
)


# ============================================================
# MISSION SUMMARY
# ============================================================

render_mission_summary(
    items,
    drawn_aoi,
    latitude,
    longitude,
    area_size,
)


# ============================================================
# MAP
# ============================================================

def map_panel_wrapper():

    state = render_map_panel(
        latitude=latitude,
        longitude=longitude,
        area_size=area_size,
        key="aoi_map",
    )

    aoi = get_selected_aoi(
        state
    )

    if aoi:

        st.session_state.drawn_aoi = aoi
        st.session_state.aoi_source = (
            "map"
        )

    elif not st.session_state.get(
        "drawn_aoi"
    ):

        st.session_state.aoi_source = (
            "manual"
        )

    return state


render_geospatial_operations_center(
    map_panel_wrapper
)


# ============================================================
# SCENE CATALOG
# ============================================================

render_scene_catalog(
    items,
    download_callback,
)


# ============================================================
# ACTIVE SCENE
# ============================================================

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
        st.session_state.get(
            "index_stats"
        ),
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

def run_change_detection(
    params,
    bbox,
) -> None:
    """
    Execute a complete Before / After comparison.
    """

    before_name = (
        params["before_name"]
    )

    after_name = (
        params["after_name"]
    )

    threshold = float(
        params["threshold"]
    )

    index_choice = (
        params["index_choice"]
    )

    scene_options = (
        params.get(
            "scene_options"
        )
        or {}
    )

    before_item = (
        scene_options.get(
            before_name
        )
    )

    after_item = (
        scene_options.get(
            after_name
        )
    )

    # --------------------------------------------------------
    # SCENE VALIDATION
    # --------------------------------------------------------

    if (
        before_item is None
        or after_item is None
    ):

        st.warning(
            "Both Before and After scenes "
            "must be selected."
        )

        return

    if (
        before_item.id
        == after_item.id
    ):

        st.warning(
            "Before and After must be "
            "different Sentinel-2 scenes."
        )

        return

    try:

        # ----------------------------------------------------
        # DOWNLOAD BEFORE
        # ----------------------------------------------------

        before_data = _download_scene(
            before_item,
            label="Before scene",
        )

        # ----------------------------------------------------
        # DOWNLOAD AFTER
        # ----------------------------------------------------

        after_data = _download_scene(
            after_item,
            label="After scene",
        )

        # ----------------------------------------------------
        # READ BEFORE
        # ----------------------------------------------------

        b04_before, m04_before = (
            read_band(
                before_data["bands"]["B04"]
            )
        )

        b03_before, m03_before = (
            read_band(
                before_data["bands"]["B03"]
            )
        )

        b08_before, m08_before = (
            read_band(
                before_data["bands"]["B08"]
            )
        )

        b11_before, m11_before = (
            read_band(
                before_data["bands"]["B11"]
            )
        )

        # ----------------------------------------------------
        # READ AFTER
        # ----------------------------------------------------

        b04_after, m04_after = (
            read_band(
                after_data["bands"]["B04"]
            )
        )

        b03_after, m03_after = (
            read_band(
                after_data["bands"]["B03"]
            )
        )

        b08_after, m08_after = (
            read_band(
                after_data["bands"]["B08"]
            )
        )

        b11_after, m11_after = (
            read_band(
                after_data["bands"]["B11"]
            )
        )

        # ----------------------------------------------------
        # VALIDATE REFERENCE RASTERS
        # ----------------------------------------------------

        validate_raster(
            b04_before,
            m04_before,
            label="Before B04",
        )

        validate_raster(
            b04_after,
            m04_after,
            label="After B04",
        )

        # ----------------------------------------------------
        # ALIGN BEFORE
        # ----------------------------------------------------

        b03_before = (
            align_band_to_reference(
                b03_before,
                m03_before,
                b04_before,
                m04_before,
            )
        )

        b08_before = (
            align_band_to_reference(
                b08_before,
                m08_before,
                b04_before,
                m04_before,
            )
        )

        b11_before = (
            align_band_to_reference(
                b11_before,
                m11_before,
                b04_before,
                m04_before,
            )
        )

        # ----------------------------------------------------
        # ALIGN AFTER
        # ----------------------------------------------------

        b03_after = (
            align_band_to_reference(
                b03_after,
                m03_after,
                b04_after,
                m04_after,
            )
        )

        b08_after = (
            align_band_to_reference(
                b08_after,
                m08_after,
                b04_after,
                m04_after,
            )
        )

        b11_after = (
            align_band_to_reference(
                b11_after,
                m11_after,
                b04_after,
                m04_after,
            )
        )

        # ----------------------------------------------------
        # CALCULATE INDEX
        # ----------------------------------------------------

        if "NDVI" in index_choice:

            before_idx = calculate_ndvi(
                red=b04_before,
                nir=b08_before,
            )

            after_idx = calculate_ndvi(
                red=b04_after,
                nir=b08_after,
            )

        elif "NDWI" in index_choice:

            before_idx = calculate_ndwi(
                green=b03_before,
                nir=b08_before,
            )

            after_idx = calculate_ndwi(
                green=b03_after,
                nir=b08_after,
            )

        else:

            before_idx = calculate_ndbi(
                nir=b08_before,
                swir=b11_before,
            )

            after_idx = calculate_ndbi(
                nir=b08_after,
                swir=b11_after,
            )

        # ----------------------------------------------------
        # DIFFERENCE
        # ----------------------------------------------------

        diff = calculate_difference(
            before=before_idx,
            after=after_idx,
            before_metadata=m04_before,
            after_metadata=m04_after,
        )

        # ----------------------------------------------------
        # CHANGE MASK
        # ----------------------------------------------------

        change_map = detect_change(
            diff,
            threshold=threshold,
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        stats = (
            calculate_change_statistics(
                change_map,
                pixel_size_meters=10.0,
            )
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig = create_change_figure(
            change_map,
            title=(
                f"{index_choice} "
                "Change Detection"
            ),
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.session_state.change_result = {
            "statistics": stats,
            "figure": fig,
            "index_name": index_choice,
            "threshold": threshold,
            "before_scene": before_item.id,
            "after_scene": after_item.id,
            "before_date": (
                str(
                    before_item.datetime.date()
                )
                if before_item.datetime
                else "Unknown"
            ),
            "after_date": (
                str(
                    after_item.datetime.date()
                )
                if after_item.datetime
                else "Unknown"
            ),
        }

        update_pipeline_status(
            "Change",
            "done",
        )

        st.success(
            "Change detection completed."
        )

    except RasterValidationError as error:

        st.error(
            "Change detection stopped because "
            "the rasters are not safely aligned."
        )

        st.warning(
            str(error)
        )

    except Exception as error:

        st.error(
            "Change detection failed."
        )

        st.exception(error)


with tab_change:

    change_params = (
        render_change_detection_controls(
            items
        )
    )

    if change_params is not None:

        bbox = _current_bbox()

        run_change_detection(
            change_params,
            bbox,
        )

        st.rerun()

    render_change_detection_results()


# ============================================================
# GEOSPATIAL AI
# ============================================================

def run_ai_inference(
    params,
    detection_rgb,
) -> None:
    """
    Run object detection on the active RGB scene.
    """

    if detection_rgb is None:

        st.warning(
            "RGB image for AI is not available."
        )

        return

    try:

        # ----------------------------------------------------
        # RESET PREVIOUS DETECTIONS
        # ----------------------------------------------------

        st.session_state.object_detections = []
        st.session_state.detection_figure = None

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        detector = SatelliteDetector(
            model_id=params["model_id"],
            device="cpu",
        )

        # ----------------------------------------------------
        # TILING
        # ----------------------------------------------------

        tiles = create_tiles(
            detection_rgb,
            tile_size=params[
                "tile_size"
            ],
            overlap=params[
                "overlap"
            ],
        )

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        detections = (
            detector.predict_tiles(
                tiles,
                confidence=params[
                    "confidence"
                ],
            )
        )

        # ----------------------------------------------------
        # FILTER CONFIDENCE
        # ----------------------------------------------------

        detections = filter_detections(
            detections,
            params["confidence"],
        )

        # ----------------------------------------------------
        # FILTER CLASSES
        # ----------------------------------------------------

        detections = filter_classes(
            detections,
            params["classes"],
        )

        st.session_state.object_detections = (
            detections
        )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        if detections:

            st.session_state.detection_figure = (
                draw_detections(
                    detection_rgb,
                    detections,
                )
            )

        update_pipeline_status(
            "AI",
            "done",
        )

        st.success(
            "AI inference completed. "
            f"{len(detections)} objects detected."
        )

    except Exception as error:

        st.error(
            "AI inference failed."
        )

        st.exception(error)


with tab_ai:

    ai_params = (
        render_geospatial_ai_controls(
            st.session_state.satellite_data,
            st.session_state.detection_rgb,
        )
    )

    if ai_params is not None:

        run_ai_inference(
            ai_params,
            st.session_state.detection_rgb,
        )

        st.rerun()

    render_geospatial_ai_results(
        st.session_state.object_detections,
        st.session_state.detection_rgb,
    )


# ============================================================
# GEOJSON EXPORT
# ============================================================

if st.session_state.get(
    "export_geojson",
    False,
):

    detections = (
        st.session_state.object_detections
    )

    transform = (
        st.session_state.get(
            "transform"
        )
    )

    crs = (
        st.session_state.get(
            "crs"
        )
    )

    if (
        detections
        and transform is not None
        and crs is not None
    ):

        try:

            gdf = (
                georeference_detections(
                    detections,
                    transform=transform,
                    crs=crs,
                )
            )

            geojson_bytes = (
                to_geojson_bytes(
                    gdf
                )
            )

            st.download_button(
                "Download GeoJSON",
                data=geojson_bytes,
                file_name=(
                    "detections.geojson"
                ),
                mime=(
                    "application/geo+json"
                ),
                key="download_geojson",
            )

        except Exception as error:

            st.error(
                "GeoJSON export failed."
            )

            st.exception(error)

    elif not detections:

        st.warning(
            "No detections are available "
            "for export."
        )

    else:

        st.warning(
            "Georeferencing metadata is "
            "not available."
        )

    st.session_state[
        "export_geojson"
    ] = False


# ============================================================
# PIPELINE STATUS
# ============================================================

render_pipeline_status()


# ============================================================
# FOOTER
# ============================================================

render_footer()