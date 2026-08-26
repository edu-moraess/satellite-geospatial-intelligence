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
from typing import Any

import numpy as np
import streamlit as st

from src.catalog import search_sentinel, create_bbox
from src.config import RAW_DIR
from src.downloader import download_required_bands
from src.geospatial import read_band, align_band_to_reference
from src.raster_validation import validate_raster
from src.visualization import create_rgb, create_false_color
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
# SESSION STATE DEFAULTS
# ============================================================

_DEFAULTS: dict[str, Any] = {
    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------
    "aoi_latitude": -23.5505,
    "aoi_longitude": -46.6333,
    "aoi_area_size": 0.05,
    "aoi_source": "manual",

    # Persistent AOI drawn on map.
    "drawn_aoi": None,

    # Last map click used to update manual coordinates.
    "_last_processed_map_click": None,

    # --------------------------------------------------------
    # Catalog / imagery
    # --------------------------------------------------------
    "search_results": [],
    "satellite_data": None,

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    "rgb_img": None,
    "false_color_img": None,

    # --------------------------------------------------------
    # Spectral
    # --------------------------------------------------------
    "ndvi": None,
    "ndwi": None,
    "ndbi": None,
    "index_stats": {},
    "index_figure": None,

    # --------------------------------------------------------
    # Land cover
    # --------------------------------------------------------
    "classification_fig": None,
    "percentages": None,
    "area_data": None,

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------
    "detection_rgb": None,
    "object_detections": [],
    "detection_figure": None,

    # --------------------------------------------------------
    # Change detection
    # --------------------------------------------------------
    "change_result": None,

    # --------------------------------------------------------
    # Georeferencing
    # --------------------------------------------------------
    "transform": None,
    "crs": None,

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------
    "export_geojson": False,
}


for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# STATE HELPERS
# ============================================================

def _reset_scene_analysis() -> None:
    """
    Reset all analysis products when a new search is executed.

    AOI and catalog state are intentionally preserved.
    """

    st.session_state.satellite_data = None

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

    st.session_state.change_result = None

    st.session_state.transform = None
    st.session_state.crs = None

    st.session_state.export_geojson = False


def _reset_downstream_after_download() -> None:
    """
    Reset products that depend on the active scene.
    """

    st.session_state.object_detections = []
    st.session_state.detection_figure = None
    st.session_state.change_result = None
    st.session_state.export_geojson = False


def _current_aoi_center() -> tuple[float, float]:
    return (
        float(st.session_state["aoi_latitude"]),
        float(st.session_state["aoi_longitude"]),
    )


def _current_area_size() -> float:
    return float(
        st.session_state["aoi_area_size"]
    )


def _current_search_bbox():
    """
    Return the drawn AOI bbox when available.

    Otherwise use the manual latitude/longitude AOI.
    """

    drawn_aoi = st.session_state.get(
        "drawn_aoi"
    )

    if drawn_aoi:
        bbox = drawn_aoi.get("bbox")

        if bbox:
            return bbox

    latitude, longitude = _current_aoi_center()
    area_size = _current_area_size()

    return create_bbox(
        latitude,
        longitude,
        area_size,
    )


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# ANALYSIS CONTROL — SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("**Analysis Control**")

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
        step=0.01,
        key="aoi_area_size",
        help="Approximate AOI side length in degrees.",
    )

    st.caption(
        "Coordinates can be entered manually or "
        "selected directly on the map."
    )

    # --------------------------------------------------------
    # TEMPORAL WINDOW
    # --------------------------------------------------------

    st.markdown("Temporal window")

    start_date = st.date_input(
        "Start",
        value=date(2026, 1, 1),
        key="analysis_start_date",
    )

    end_date = st.date_input(
        "End",
        value=date(2026, 8, 23),
        key="analysis_end_date",
    )

    # --------------------------------------------------------
    # CLOUD FILTER
    # --------------------------------------------------------

    st.markdown("Scene filter")

    max_cloud_cover = st.slider(
        "Max cloud cover",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        format="%d%%",
        key="max_cloud_cover",
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_clicked = st.button(
        "Search Sentinel-2",
        type="primary",
        use_container_width=True,
        key="search_sentinel_btn",
    )


# ============================================================
# CURRENT AOI VALUES
# ============================================================

latitude = float(
    st.session_state["aoi_latitude"]
)

longitude = float(
    st.session_state["aoi_longitude"]
)

area_size = float(
    st.session_state["aoi_area_size"]
)


# ============================================================
# SENTINEL-2 SEARCH
# ============================================================

if search_clicked:

    if start_date > end_date:

        st.error(
            "Start date must be before end date."
        )

    else:

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
                    start_date=str(start_date),
                    end_date=str(end_date),
                    max_cloud_cover=max_cloud_cover,
                    bbox=search_bbox,
                )

                st.session_state.search_results = (
                    results
                )

                _reset_scene_analysis()

                update_pipeline_status(
                    "Catalog",
                    "done",
                )

            except Exception as e:

                st.error(
                    "Satellite catalog search failed."
                )

                st.exception(e)


# ============================================================
# CURRENT DATA
# ============================================================

items = st.session_state.get(
    "search_results",
    [],
)

drawn_aoi = st.session_state.get(
    "drawn_aoi"
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

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT clear the previously selected AOI merely because
    # the map component returned no drawings during a rerun.
    #
    # This prevents Streamlit/Folium state synchronization
    # from destroying the AOI.
    # --------------------------------------------------------

    if state:

        selected_aoi = (
            get_selected_aoi(state)
        )

        if selected_aoi is not None:

            st.session_state.drawn_aoi = (
                selected_aoi
            )

            st.session_state.aoi_source = (
                "map"
            )

    return state


render_geospatial_operations_center(
    map_panel_wrapper
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
# BAND PROCESSING
# ============================================================

def _process_bands() -> bool:
    """
    Process the active Sentinel-2 scene.

    Returns:
        True  -> processing succeeded
        False -> processing failed
    """

    data = (
        st.session_state.get(
            "satellite_data"
        )
    )

    if data is None:
        return False

    try:

        # ----------------------------------------------------
        # READ BANDS
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
        # ALIGN TO B04 REFERENCE GRID
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
        # RASTER VALIDATION
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

        ndvi_s = _index_stats(
            ndvi
        )

        ndwi_s = _index_stats(
            ndwi
        )

        ndbi_s = _index_stats(
            ndbi
        )

        st.session_state.ndvi = (
            ndvi_s["mean"]
            if ndvi_s
            else None
        )

        st.session_state.ndwi = (
            ndwi_s["mean"]
            if ndwi_s
            else None
        )

        st.session_state.ndbi = (
            ndbi_s["mean"]
            if ndbi_s
            else None
        )

        st.session_state.index_stats = {
            "ndvi": ndvi_s,
            "ndwi": ndwi_s,
            "ndbi": ndbi_s,
        }

        # ----------------------------------------------------
        # INDEX VISUALIZATION
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

        return True

    except Exception as e:

        st.error(
            "Failed to process satellite bands."
        )

        st.exception(e)

        return False


# ============================================================
# DOWNLOAD CALLBACK
# ============================================================

def download_callback(item) -> None:

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

    drawn_aoi = (
        st.session_state.get(
            "drawn_aoi"
        )
    )

    if drawn_aoi:

        bbox = drawn_aoi.get(
            "bbox"
        )

    else:

        bbox = create_bbox(
            current_latitude,
            current_longitude,
            current_area_size,
        )

    with st.spinner(
        f"Downloading {item.id}..."
    ):

        try:

            bands = (
                download_required_bands(
                    item=item,
                    bbox=bbox,
                    output_directory=(
                        RAW_DIR / item.id
                    ),
                )
            )

            st.session_state.satellite_data = {
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
            }

            # Reset only downstream products.
            _reset_downstream_after_download()

            processed = _process_bands()

            if processed:

                update_pipeline_status(
                    "Imagery",
                    "done",
                )

                update_pipeline_status(
                    "Spectral",
                    "done",
                )

                st.success(
                    "Scene downloaded and processed."
                )

        except Exception as e:

            st.error(
                "Download failed."
            )

            st.exception(e)


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
    st.session_state.get(
        "satellite_data"
    ),
    st.session_state.get(
        "rgb_img"
    ),
    st.session_state.get(
        "false_color_img"
    ),
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
        st.session_state.get(
            "ndvi"
        ),
        st.session_state.get(
            "ndwi"
        ),
        st.session_state.get(
            "ndbi"
        ),
        st.session_state.get(
            "index_figure"
        ),
        st.session_state.get(
            "index_stats"
        ),
    )


# ============================================================
# LAND COVER
# ============================================================

with tab_land:

    render_land_cover(
        st.session_state.get(
            "classification_fig"
        ),
        st.session_state.get(
            "percentages"
        ),
        st.session_state.get(
            "area_data"
        ),
    )


# ============================================================
# CHANGE DETECTION
# ============================================================

def run_change_detection(
    params,
    bbox,
) -> None:

    before_name = params[
        "before_name"
    ]

    after_name = params[
        "after_name"
    ]

    threshold = params[
        "threshold"
    ]

    index_choice = params[
        "index_choice"
    ]

    scene_options = (
        params.get(
            "scene_options"
        )
        or {}
    )

    if not scene_options:

        for it in items:

            date_str = (
                str(
                    it.datetime.date()
                )
                if it.datetime
                else "Unknown"
            )

            cloud = float(
                it.properties.get(
                    "eo:cloud_cover",
                    0,
                )
            )

            label = (
                f"{date_str} · "
                f"{cloud:.2f}% · "
                f"{it.id[:12]}"
            )

            scene_options[
                label
            ] = it

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

    if (
        before_item is None
        or after_item is None
        or before_item.id
        == after_item.id
    ):

        st.warning(
            "Choose two different scenes."
        )

        return

    try:

        # ----------------------------------------------------
        # BEFORE
        # ----------------------------------------------------

        with st.spinner(
            "Downloading Data A..."
        ):

            before_bands = (
                download_required_bands(
                    item=before_item,
                    bbox=bbox,
                    output_directory=(
                        RAW_DIR
                        / before_item.id
                    ),
                )
            )

        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------

        with st.spinner(
            "Downloading Data B..."
        ):

            after_bands = (
                download_required_bands(
                    item=after_item,
                    bbox=bbox,
                    output_directory=(
                        RAW_DIR
                        / after_item.id
                    ),
                )
            )

        # ----------------------------------------------------
        # READ BEFORE
        # ----------------------------------------------------

        b04_b, m04_b = read_band(
            before_bands["B04"]
        )

        b03_b, m03_b = read_band(
            before_bands["B03"]
        )

        b08_b, m08_b = read_band(
            before_bands["B08"]
        )

        b11_b, m11_b = read_band(
            before_bands["B11"]
        )

        # ----------------------------------------------------
        # READ AFTER
        # ----------------------------------------------------

        b04_a, m04_a = read_band(
            after_bands["B04"]
        )

        b03_a, m03_a = read_band(
            after_bands["B03"]
        )

        b08_a, m08_a = read_band(
            after_bands["B08"]
        )

        b11_a, m11_a = read_band(
            after_bands["B11"]
        )

        # ----------------------------------------------------
        # ALIGN BEFORE
        # ----------------------------------------------------

        b03_b = align_band_to_reference(
            b03_b,
            m03_b,
            b04_b,
            m04_b,
        )

        b08_b = align_band_to_reference(
            b08_b,
            m08_b,
            b04_b,
            m04_b,
        )

        b11_b = align_band_to_reference(
            b11_b,
            m11_b,
            b04_b,
            m04_b,
        )

        # ----------------------------------------------------
        # ALIGN AFTER
        # ----------------------------------------------------

        b03_a = align_band_to_reference(
            b03_a,
            m03_a,
            b04_a,
            m04_a,
        )

        b08_a = align_band_to_reference(
            b08_a,
            m08_a,
            b04_a,
            m04_a,
        )

        b11_a = align_band_to_reference(
            b11_a,
            m11_a,
            b04_a,
            m04_a,
        )

        # ----------------------------------------------------
        # INDEX
        # ----------------------------------------------------

        if "NDVI" in index_choice:

            before_idx = calculate_ndvi(
                b04_b,
                b08_b,
            )

            after_idx = calculate_ndvi(
                b04_a,
                b08_a,
            )

        elif "NDWI" in index_choice:

            before_idx = calculate_ndwi(
                b03_b,
                b08_b,
            )

            after_idx = calculate_ndwi(
                b03_a,
                b08_a,
            )

        else:

            before_idx = calculate_ndbi(
                b08_b,
                b11_b,
            )

            after_idx = calculate_ndbi(
                b08_a,
                b11_a,
            )

        # ----------------------------------------------------
        # DIFFERENCE
        # ----------------------------------------------------

        diff = calculate_difference(
            before_idx,
            after_idx,
            before_metadata=m04_b,
            after_metadata=m04_a,
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
        # VISUALIZATION
        # ----------------------------------------------------

        fig = create_change_figure(
            change_map,
            title=(
                f"{index_choice} "
                "Change Detection"
            ),
        )

        st.session_state.change_result = {
            "statistics": stats,
            "figure": fig,
            "index_name": index_choice,
            "threshold": threshold,
            "before_scene": before_item.id,
            "after_scene": after_item.id,
        }

        update_pipeline_status(
            "Change",
            "done",
        )

        st.success(
            "Change detection completed."
        )

    except Exception as e:

        st.error(
            "Change detection failed."
        )

        st.exception(e)


with tab_change:

    change_params = (
        render_change_detection_controls(
            items
        )
    )

    if change_params is not None:

        bbox = _current_search_bbox()

        run_change_detection(
            change_params,
            bbox,
        )

    render_change_detection_results()


# ============================================================
# GEOSPATIAL AI
# ============================================================

def run_ai_inference(
    params,
    detection_rgb,
) -> None:

    if detection_rgb is None:

        st.warning(
            "RGB image for AI is not available."
        )

        return

    try:

        detector = SatelliteDetector(
            model_id=params[
                "model_id"
            ],
            device="cpu",
        )

        tiles = create_tiles(
            detection_rgb,
            tile_size=params[
                "tile_size"
            ],
            overlap=params[
                "overlap"
            ],
        )

        detections = (
            detector.predict_tiles(
                tiles,
                confidence=params[
                    "confidence"
                ],
            )
        )

        detections = filter_detections(
            detections,
            params[
                "confidence"
            ],
        )

        detections = filter_classes(
            detections,
            params[
                "classes"
            ],
        )

        st.session_state.object_detections = (
            detections
        )

        if detections:

            st.session_state.detection_figure = (
                draw_detections(
                    detection_rgb,
                    detections,
                )
            )

        else:

            st.session_state.detection_figure = (
                None
            )

        update_pipeline_status(
            "AI",
            "done",
        )

        st.success(
            "AI inference completed. "
            f"{len(detections)} objects detected."
        )

    except Exception as e:

        st.error(
            "AI inference failed."
        )

        st.exception(e)


with tab_ai:

    ai_params = (
        render_geospatial_ai_controls(
            st.session_state.get(
                "satellite_data"
            ),
            st.session_state.get(
                "detection_rgb"
            ),
        )
    )

    if ai_params is not None:

        run_ai_inference(
            ai_params,
            st.session_state.get(
                "detection_rgb"
            ),
        )

    render_geospatial_ai_results(
        st.session_state.get(
            "object_detections",
            [],
        ),
        st.session_state.get(
            "detection_rgb"
        ),
    )


# ============================================================
# GEOJSON EXPORT
# ============================================================

if st.session_state.get(
    "export_geojson",
    False,
):

    detections = (
        st.session_state.get(
            "object_detections",
            [],
        )
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

            gdf = georeference_detections(
                detections,
                transform=transform,
                crs=crs,
            )

            geojson_bytes = (
                to_geojson_bytes(
                    gdf
                )
            )

            st.download_button(
                "Download GeoJSON",
                data=geojson_bytes,
                file_name="detections.geojson",
                mime="application/geo+json",
                key="download_geojson",
            )

        except Exception as e:

            st.error(
                "GeoJSON export failed."
            )

            st.exception(e)

    elif not detections:

        st.warning(
            "There are no detections to export."
        )

    elif transform is None:

        st.warning(
            "Georeferencing transform is unavailable."
        )

    elif crs is None:

        st.warning(
            "Coordinate reference system is unavailable."
        )

    # Prevent repeating the export action.
    st.session_state[
        "export_geojson"
    ] = False


# ============================================================
# PIPELINE
# ============================================================

render_pipeline_status()


# ============================================================
# FOOTER
# ============================================================

render_footer()