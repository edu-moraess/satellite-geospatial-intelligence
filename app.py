"""
app.py — Satellite Geospatial Intelligence

Orchestration only:
- page configuration
- session state
- AOI control
- catalog search
- scene download
- scientific pipeline wiring
- change detection
- AI inference
- export

Presentation lives in ui/.
Scientific logic lives in src/.
"""

from __future__ import annotations

from datetime import date

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
# SESSION DEFAULTS
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
    # Catalog
    # --------------------------------------------------------

    "search_results": [],
    "drawn_aoi": None,

    # --------------------------------------------------------
    # Active scene
    # --------------------------------------------------------

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
    # Change detection
    # --------------------------------------------------------

    "change_result": None,

    # --------------------------------------------------------
    # Geospatial AI
    # --------------------------------------------------------

    "detection_rgb": None,
    "object_detections": [],
    "detection_figure": None,

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
# AOI HELPERS
# ============================================================

def _current_aoi_center() -> tuple[float, float]:
    """
    Return the current AOI center from session state.
    """

    return (
        float(
            st.session_state[
                "aoi_latitude"
            ]
        ),
        float(
            st.session_state[
                "aoi_longitude"
            ]
        ),
    )


def _current_area_size() -> float:
    """
    Return current AOI area size.
    """

    return float(
        st.session_state[
            "aoi_area_size"
        ]
    )


def _current_bbox():
    """
    Return the active search/analysis bounding box.

    Drawn AOI takes precedence over the manual AOI.
    """

    drawn_aoi = (
        st.session_state.get(
            "drawn_aoi"
        )
    )

    if drawn_aoi:

        bbox = drawn_aoi.get(
            "bbox"
        )

        if bbox:

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
# STATE RESET HELPERS
# ============================================================

def _reset_scene_analysis() -> None:
    """
    Reset all analysis products associated with the active
    satellite scene.

    This prevents results from an old scene remaining visible
    after a new scene is downloaded.
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


def _reset_change_result() -> None:
    """
    Reset only change-detection state.
    """

    st.session_state.change_result = None


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# ANALYSIS CONTROL — SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "**Analysis Control**"
    )

    st.caption(
        "AOI · temporal window · scene filter"
    )

    st.markdown(
        "AOI"
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        key="aoi_latitude",
        format="%.6f",
        help=(
            "Analysis area center latitude."
        ),
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        key="aoi_longitude",
        format="%.6f",
        help=(
            "Analysis area center longitude."
        ),
    )

    area_size = st.slider(
        "Area size (deg)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        key="aoi_area_size",
        help=(
            "Approximate AOI side length in degrees."
        ),
    )

    st.caption(
        "Coordinates can be entered manually or "
        "selected directly on the map."
    )

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

    search_clicked = st.button(
        "Search Sentinel-2",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# AOI VALUES AFTER WIDGET STATE UPDATE
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

            st.session_state.satellite_data = (
                None
            )

            _reset_scene_analysis()

            update_pipeline_status(
                "Catalog",
                "done",
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Satellite catalog search failed."
            )

            st.exception(
                error
            )


# ============================================================
# CURRENT CATALOG DATA
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

        st.session_state.drawn_aoi = (
            aoi
        )

        st.session_state.aoi_source = (
            "map"
        )

    else:

        st.session_state.drawn_aoi = None

        st.session_state.aoi_source = (
            "manual"
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

def _process_bands() -> None:

    data = (
        st.session_state.satellite_data
    )

    if data is None:
        return

    try:

        # ----------------------------------------------------
        # READ NATIVE BANDS
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
        # VALIDATE REFERENCE BAND
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
        # VALIDATE FINAL GRID
        # ----------------------------------------------------

        for band, label in [
            (b02, "B02"),
            (b03, "B03"),
            (b04, "B04"),
            (b08, "B08"),
            (b11, "B11"),
        ]:

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

        st.session_state.rgb_img = (
            rgb
        )

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
        # GEOSPATIAL METADATA
        # ----------------------------------------------------

        st.session_state.transform = (
            m04["transform"]
        )

        st.session_state.crs = str(
            m04["crs"]
        )

    except RasterValidationError as error:

        st.error(
            "Raster validation failed."
        )

        st.warning(
            str(error)
        )

        update_pipeline_status(
            "Spectral",
            "error",
        )

    except Exception as error:

        st.error(
            "Failed to process satellite bands."
        )

        st.exception(
            error
        )

        update_pipeline_status(
            "Spectral",
            "error",
        )


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

    bbox = _current_bbox()

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

            # ----------------------------------------------
            # RESET PREVIOUS SCENE PRODUCTS
            # ----------------------------------------------

            _reset_scene_analysis()

            # ----------------------------------------------
            # ACTIVE SCENE
            # ----------------------------------------------

            st.session_state.satellite_data = {

                "scene_id": item.id,

                "date": (
                    str(
                        item.datetime.date()
                    )
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

                "latitude": (
                    current_latitude
                ),

                "longitude": (
                    current_longitude
                ),

                "area_size": (
                    current_area_size
                ),

                "bbox": bbox,

                "crs": None,

                "transform": None,
            }

            # ----------------------------------------------
            # PROCESS
            # ----------------------------------------------

            _process_bands()

            # ----------------------------------------------
            # PIPELINE STATUS
            # ----------------------------------------------

            update_pipeline_status(
                "Imagery",
                "done",
            )

            update_pipeline_status(
                "Spectral",
                "done",
            )

            st.rerun()

        except RasterValidationError as error:

            st.error(
                "Downloaded raster failed validation."
            )

            st.warning(
                str(error)
            )

            update_pipeline_status(
                "Imagery",
                "error",
            )

        except Exception as error:

            st.error(
                "Download failed."
            )

            st.exception(
                error
            )

            update_pipeline_status(
                "Imagery",
                "error",
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

def _build_scene_options():

    scene_options = {}

    for item in items:

        date_str = (
            str(
                item.datetime.date()
            )
            if item.datetime
            else "Unknown"
        )

        cloud = float(
            item.properties.get(
                "eo:cloud_cover",
                0,
            )
        )

        label = (
            f"{date_str} · "
            f"{cloud:.2f}% · "
            f"{item.id[:12]}"
        )

        scene_options[
            label
        ] = item

    return scene_options


def _calculate_scene_index(
    index_choice,
    b03,
    b04,
    b08,
    b11,
):
    """
    Calculate the selected spectral index.

    All inputs must already share the same spatial grid.
    """

    if "NDVI" in index_choice:

        return calculate_ndvi(
            red=b04,
            nir=b08,
        )

    if "NDWI" in index_choice:

        return calculate_ndwi(
            green=b03,
            nir=b08,
        )

    return calculate_ndbi(
        nir=b08,
        swir=b11,
    )


def _download_scene_for_change(
    item,
    bbox,
):
    """
    Download the bands required for one temporal scene.
    """

    return download_required_bands(
        item=item,
        bbox=bbox,
        output_directory=(
            RAW_DIR / item.id
        ),
    )


def _read_and_align_change_bands(
    bands,
):
    """
    Read all required bands and align them to B04.

    Returns:
        b03, b04, b08, b11, m04
    """

    b04, m04 = read_band(
        bands["B04"]
    )

    b03, m03 = read_band(
        bands["B03"]
    )

    b08, m08 = read_band(
        bands["B08"]
    )

    b11, m11 = read_band(
        bands["B11"]
    )

    validate_raster(
        b04,
        m04,
        label="B04 reference",
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

    validate_raster(
        b03,
        label="B03 aligned",
    )

    validate_raster(
        b04,
        label="B04 reference",
    )

    validate_raster(
        b08,
        label="B08 aligned",
    )

    validate_raster(
        b11,
        label="B11 aligned",
    )

    return (
        b03,
        b04,
        b08,
        b11,
        m04,
    )


def run_change_detection(
    params,
    bbox,
) -> None:

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
        or _build_scene_options()
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
    # VALIDATE SCENE SELECTION
    # --------------------------------------------------------

    if (
        before_item is None
        or after_item is None
    ):

        st.warning(
            "Could not resolve the selected scenes."
        )

        return

    if (
        before_item.id
        == after_item.id
    ):

        st.warning(
            "Choose two different scenes "
            "for temporal comparison."
        )

        return

    # --------------------------------------------------------
    # VALIDATE TEMPORAL ORDER
    # --------------------------------------------------------

    if (
        before_item.datetime
        and after_item.datetime
        and after_item.datetime
        < before_item.datetime
    ):

        st.warning(
            "The 'After' scene must be later "
            "than the 'Before' scene."
        )

        return

    try:

        # ----------------------------------------------------
        # DOWNLOAD BEFORE
        # ----------------------------------------------------

        with st.spinner(
            "Downloading Before scene..."
        ):

            before_bands = (
                _download_scene_for_change(
                    before_item,
                    bbox,
                )
            )

        # ----------------------------------------------------
        # DOWNLOAD AFTER
        # ----------------------------------------------------

        with st.spinner(
            "Downloading After scene..."
        ):

            after_bands = (
                _download_scene_for_change(
                    after_item,
                    bbox,
                )
            )

        # ----------------------------------------------------
        # READ + ALIGN BEFORE
        # ----------------------------------------------------

        (
            b03_before,
            b04_before,
            b08_before,
            b11_before,
            m04_before,
        ) = _read_and_align_change_bands(
            before_bands
        )

        # ----------------------------------------------------
        # READ + ALIGN AFTER
        # ----------------------------------------------------

        (
            b03_after,
            b04_after,
            b08_after,
            b11_after,
            m04_after,
        ) = _read_and_align_change_bands(
            after_bands
        )

        # ----------------------------------------------------
        # CALCULATE INDEX
        # ----------------------------------------------------

        before_idx = (
            _calculate_scene_index(
                index_choice,
                b03_before,
                b04_before,
                b08_before,
                b11_before,
            )
        )

        after_idx = (
            _calculate_scene_index(
                index_choice,
                b03_after,
                b04_after,
                b08_after,
                b11_after,
            )
        )

        # ----------------------------------------------------
        # TEMPORAL DIFFERENCE
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
        # VISUALIZATION
        # ----------------------------------------------------

        fig = create_change_figure(
            change_map,
            title=(
                f"{index_choice} "
                "Change Detection"
            ),
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        st.session_state.change_result = {

            "statistics": stats,

            "figure": fig,

            "index_name": (
                index_choice
            ),

            "threshold": (
                threshold
            ),

            "before_scene": (
                before_item.id
            ),

            "after_scene": (
                after_item.id
            ),

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

            "difference": diff,

            "change_map": change_map,

            "before_metadata": (
                m04_before
            ),

            "after_metadata": (
                m04_after
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

        update_pipeline_status(
            "Change",
            "error",
        )

        st.error(
            "Change detection stopped: "
            "raster alignment is unsafe."
        )

        st.warning(
            str(error)
        )

    except Exception as error:

        update_pipeline_status(
            "Change",
            "error",
        )

        st.error(
            "Change detection failed."
        )

        st.exception(
            error
        )


with tab_change:

    change_params = (
        render_change_detection_controls(
            items
        )
    )

    if change_params is not None:

        bbox = _current_bbox()

        _reset_change_result()

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

        detections = (
            filter_detections(
                detections,
                params[
                    "confidence"
                ],
            )
        )

        detections = (
            filter_classes(
                detections,
                params[
                    "classes"
                ],
            )
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

    except Exception as error:

        update_pipeline_status(
            "AI",
            "error",
        )

        st.error(
            "AI inference failed."
        )

        st.exception(
            error
        )


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
            )

        except Exception as error:

            st.error(
                "GeoJSON export failed."
            )

            st.exception(
                error
            )

    elif not detections:

        st.warning(
            "There are no detections to export."
        )

    else:

        st.warning(
            "Geospatial reference is unavailable "
            "for this scene."
        )

    st.session_state[
        "export_geojson"
    ] = False


# ============================================================
# FOOTER
# ============================================================

render_pipeline_status()

render_footer()