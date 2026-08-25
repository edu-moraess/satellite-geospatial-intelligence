from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.catalog import search_sentinel, create_bbox
from src.config import RAW_DIR
from src.downloader import download_required_bands
from src.geospatial import read_band, align_band_to_reference

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

from src.geospatial_detections import (
    georeference_detections,
    to_geojson_bytes,
)

from src.map_view import (
    render_map_panel,
)

from src.aoi import (
    get_selected_aoi,
    format_bbox,
)

from ui.theme import apply_theme

from ui.layout import (
    render_header,
    render_section,
    render_footer,
    render_operations_header,
)

from ui.components import (
    metric_card,
    kpi_card,
    status_badge,
    info_card,
)

from ui.status import (
    render_pipeline_status,
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

apply_theme()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "search_results": [],
    "satellite_data": None,
    "change_result": None,
    "object_detections": [],
    "drawn_aoi": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

render_header(
    title="Satellite Geospatial Intelligence",
    subtitle=(
        "Earth Observation • Remote Sensing • Geospatial AI"
    ),
)


# ============================================================
# SIDEBAR — CONFIGURATION
# ============================================================

st.sidebar.markdown(
    "## Mission Configuration"
)

st.sidebar.caption(
    "Define the area, temporal window and image quality "
    "before querying the Sentinel-2 catalog."
)


# ------------------------------------------------------------
# AOI
# ------------------------------------------------------------

st.sidebar.markdown(
    "### 📍 Area of Interest"
)

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=-23.5505,
    format="%.6f",
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-46.6333,
    format="%.6f",
)

area_size = st.sidebar.slider(
    "Area size",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
)


# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

st.sidebar.markdown(
    "### 📅 Temporal Window"
)

start_date = st.sidebar.date_input(
    "Start date",
    value=date(2026, 1, 1),
)

end_date = st.sidebar.date_input(
    "End date",
    value=date(2026, 8, 23),
)


# ------------------------------------------------------------
# CLOUD
# ------------------------------------------------------------

st.sidebar.markdown(
    "### ☁️ Image Quality"
)

max_cloud_cover = st.sidebar.slider(
    "Maximum cloud coverage",
    min_value=0,
    max_value=100,
    value=10,
    step=1,
    format="%d%%",
)


# ============================================================
# SEARCH
# ============================================================

if st.sidebar.button(
    "🔎 Search Satellite Data",
    type="primary",
    use_container_width=True,
):

    if start_date > end_date:

        st.error(
            "Start date must be before the end date."
        )

        st.stop()

    drawn_aoi_for_search = (
        st.session_state.get("drawn_aoi")
    )

    with st.spinner(
        "🛰️ Searching Sentinel-2 catalog..."
    ):

        try:

            results = search_sentinel(
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

            st.rerun()

        except Exception as error:

            st.error(
                "❌ Satellite catalog search failed."
            )

            st.exception(error)


# ============================================================
# GLOBAL DATA REFERENCES
# ============================================================

items = st.session_state.search_results
data = st.session_state.satellite_data

detection_rgb = None

if data is not None:
    pass


# ============================================================
# AOI OPERATIONS CENTER
# ============================================================

render_section(
    "Geospatial Operations Center",
    description=(
        "Interactive Earth observation map • "
        "Sentinel-2 • AOI • Spatial analysis"
    ),
    icon="🗺️",
)

st.markdown(
    '<div class="sgi-operations">',
    unsafe_allow_html=True,
)

render_operations_header()

aoi_map_state = render_map_panel(
    latitude=latitude,
    longitude=longitude,
    area_size=area_size,
    key="aoi_selection_map",
)

drawn_aoi = get_selected_aoi(
    aoi_map_state
)

if drawn_aoi:

    st.session_state.drawn_aoi = drawn_aoi

    aoi_col1, aoi_col2, aoi_col3 = st.columns(3)

    with aoi_col1:

        metric_card(
            "AOI Type",
            drawn_aoi["geometry_type"],
        )

    with aoi_col2:

        metric_card(
            "Approx. Area",
            f"{drawn_aoi['area_km2']:.2f} km²",
        )

    with aoi_col3:

        metric_card(
            "Bounding Box",
            format_bbox(drawn_aoi["bbox"]),
        )

    if st.button(
        "🗑️ Clear drawn AOI",
        key="clear_aoi",
    ):

        st.session_state.drawn_aoi = None
        st.rerun()

else:

    st.session_state.drawn_aoi = None

    st.caption(
        "No AOI drawn — search uses the sidebar "
        "latitude, longitude and area size."
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# TOP KPI BAR
# ============================================================

if items:

    sorted_items = sorted(
        items,
        key=lambda item: float(
            item.properties.get(
                "eo:cloud_cover",
                100,
            )
        ),
    )

    best_cloud = float(
        sorted_items[0].properties.get(
            "eo:cloud_cover",
            0,
        )
    )

    latest_dates = [
        item.datetime.date()
        for item in items
        if item.datetime
    ]

    latest_date = (
        max(latest_dates)
        if latest_dates
        else "N/A"
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        kpi_card(
            "Satellite Scenes",
            str(len(items)),
            "Sentinel-2 catalog",
        )

    with k2:
        kpi_card(
            "Best Cloud Cover",
            f"{best_cloud:.2f}%",
            "Lowest scene cloud",
        )

    with k3:
        kpi_card(
            "Latest Observation",
            str(latest_date),
            "Most recent scene",
        )

    with k4:
        kpi_card(
            "AOI Status",
            "ACTIVE" if drawn_aoi else "DEFAULT",
            "Analysis geometry",
        )


# ============================================================
# MAIN WORKSPACE TABS
# ============================================================

overview_tab, change_tab, ai_tab = st.tabs(
    [
        "🌍 Overview",
        "🔬 Change Detection",
        "🎯 Geospatial AI",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    # --------------------------------------------------------
    # SATELLITE ARCHIVE
    # --------------------------------------------------------

    render_section(
        "Satellite Archive",
        description=(
            "Available Sentinel-2 observations "
            "ordered by acquisition quality"
        ),
        icon="🛰️",
    )

    if not items:

        st.info(
            "Search the Sentinel-2 catalog from the sidebar "
            "to populate the satellite archive."
        )

    else:

        archive_rows = []

        for index, item in enumerate(items):

            cloud = float(
                item.properties.get(
                    "eo:cloud_cover",
                    0,
                )
            )

            acquisition_date = (
                item.datetime.date()
                if item.datetime
                else "Unknown"
            )

            if cloud <= 1:
                quality = "Excellent"
            elif cloud <= 5:
                quality = "Good"
            elif cloud <= 10:
                quality = "Acceptable"
            else:
                quality = "Cloudy"

            archive_rows.append(
                {
                    "Date": str(acquisition_date),
                    "Cloud": f"{cloud:.2f}%",
                    "Quality": quality,
                    "Scene": item.id,
                    "_index": index,
                }
            )

        archive_display = pd.DataFrame(
            archive_rows
        ).drop(
            columns=["_index"]
        )

        st.dataframe(
            archive_display.head(10),
            use_container_width=True,
            hide_index=True,
        )

        if len(items) > 10:

            st.caption(
                f"Showing 10 of {len(items)} scenes. "
                "Use the selector below to inspect any scene."
            )

        scene_labels = []

        for index, item in enumerate(items):

            cloud = float(
                item.properties.get(
                    "eo:cloud_cover",
                    0,
                )
            )

            acquisition_date = (
                item.datetime.date()
                if item.datetime
                else "Unknown"
            )

            scene_labels.append(
                (
                    f"{acquisition_date} • "
                    f"{cloud:.2f}% clouds • "
                    f"{item.id}"
                )
            )

        selected_scene_label = st.selectbox(
            "Select scene",
            scene_labels,
            key="overview_scene_selector",
        )

        selected_item = items[
            scene_labels.index(
                selected_scene_label
            )
        ]

        selected_cloud = float(
            selected_item.properties.get(
                "eo:cloud_cover",
                0,
            )
        )

        selected_date = (
            selected_item.datetime.date()
            if selected_item.datetime
            else "Unknown"
        )

        sc1, sc2, sc3 = st.columns(3)

        with sc1:

            metric_card(
                "Acquisition",
                str(selected_date),
            )

        with sc2:

            metric_card(
                "Cloud Coverage",
                f"{selected_cloud:.2f}%",
            )

        with sc3:

            metric_card(
                "Scene ID",
                selected_item.id[:24],
            )

        if st.button(
            "⬇️ Download & Analyze Selected Scene",
            type="primary",
            use_container_width=True,
            key="download_selected_scene",
        ):

            bbox = (
                drawn_aoi["bbox"]
                if drawn_aoi
                else create_bbox(
                    latitude=latitude,
                    longitude=longitude,
                    area_size=area_size,
                )
            )

            output_directory = (
                RAW_DIR / selected_item.id
            )

            with st.spinner(
                "⬇️ Downloading Sentinel-2 data..."
            ):

                try:

                    downloaded = (
                        download_required_bands(
                            item=selected_item,
                            bbox=bbox,
                            output_directory=(
                                output_directory
                            ),
                        )
                    )

                    st.session_state.satellite_data = {
                        "scene_id": selected_item.id,
                        "date": str(selected_date),
                        "cloud": selected_cloud,
                        "bands": downloaded,
                        "latitude": latitude,
                        "longitude": longitude,
                        "area_size": area_size,
                    }

                    st.session_state.change_result = None
                    st.session_state.object_detections = []

                    st.success(
                        "✅ Satellite scene downloaded."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Satellite download failed."
                    )

                    st.exception(error)


    # --------------------------------------------------------
    # SELECTED SCENE
    # --------------------------------------------------------

    data = st.session_state.satellite_data

    if data is None:

        st.info(
            "Download a satellite scene to activate "
            "spectral analysis, land-cover classification "
            "and visualization."
        )

    else:

        render_section(
            "Active Scene",
            description=(
                "Current Sentinel-2 dataset loaded "
                "into the analytical engine"
            ),
            icon="🛰️",
        )

        ac1, ac2, ac3 = st.columns(3)

        with ac1:
            metric_card(
                "Acquisition",
                data["date"],
            )

        with ac2:
            metric_card(
                "Cloud Coverage",
                f"{data['cloud']:.2f}%",
            )

        with ac3:
            metric_card(
                "Scene",
                data["scene_id"][:24],
            )


        # ----------------------------------------------------
        # LOAD BANDS
        # ----------------------------------------------------

        with st.spinner(
            "📡 Loading spectral bands..."
        ):

            try:

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

            except Exception as error:

                st.error(
                    "❌ Failed to load satellite bands."
                )

                st.exception(error)
                st.stop()


        # ----------------------------------------------------
        # ALIGN BANDS
        # ----------------------------------------------------

        with st.spinner(
            "🔄 Aligning spectral grids..."
        ):

            try:

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

            except Exception as error:

                st.error(
                    "❌ Failed to align spectral bands."
                )

                st.exception(error)
                st.stop()


        # ----------------------------------------------------
        # RASTER VALIDATION
        # ----------------------------------------------------

        try:

            for band_label, band_array in (
                ("Blue (B02)", b02),
                ("Green (B03)", b03),
                ("Red (B04)", b04),
                ("NIR (B08)", b08),
                ("SWIR (B11)", b11),
            ):

                validate_raster(
                    band_array,
                    label=band_label,
                )

        except RasterValidationError as error:

            st.error(
                "❌ Raster invalid — spectral processing "
                "cannot continue safely."
            )

            st.caption(str(error))
            st.stop()


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        try:

            rgb = create_rgb(
                blue=b02,
                green=b03,
                red=b04,
            )

        except Exception as error:

            st.error(
                "❌ Failed to create RGB image."
            )

            st.exception(error)
            st.stop()


        # ----------------------------------------------------
        # AI RGB
        # ----------------------------------------------------

        try:

            detection_rgb = normalize_rgb(
                red=b04,
                green=b03,
                blue=b02,
            )

            validate_detection_image(
                detection_rgb
            )

        except Exception:

            detection_rgb = None


        # ----------------------------------------------------
        # FALSE COLOR
        # ----------------------------------------------------

        try:

            false_color = create_false_color(
                green=b03,
                red=b04,
                nir=b08,
            )

        except Exception as error:

            st.error(
                "❌ Failed to create False Color."
            )

            st.exception(error)
            st.stop()


        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        render_section(
            "Satellite Visualization",
            description=(
                "Natural color and false-color composites"
            ),
            icon="🌍",
        )

        image1, image2 = st.columns(2)

        with image1:

            st.image(
                rgb,
                caption="Sentinel-2 Natural Color",
                use_container_width=True,
            )

        with image2:

            st.image(
                false_color,
                caption="Sentinel-2 False Color",
                use_container_width=True,
            )


        # ----------------------------------------------------
        # MULTISPECTRAL
        # ----------------------------------------------------

        render_section(
            "Multispectral Analysis",
            description="NDVI • NDWI • NDBI",
            icon="🔬",
        )

        try:

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

        except Exception as error:

            st.error(
                "❌ Failed to calculate spectral indices."
            )

            st.exception(error)
            st.stop()


        valid_ndvi = ndvi[
            np.isfinite(ndvi)
        ]

        valid_ndwi = ndwi[
            np.isfinite(ndwi)
        ]

        valid_ndbi = ndbi[
            np.isfinite(ndbi)
        ]


        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            metric_card(
                "Mean NDVI",
                (
                    f"{np.mean(valid_ndvi):.3f}"
                    if valid_ndvi.size
                    else "N/A"
                ),
            )

        with metric2:

            metric_card(
                "Mean NDWI",
                (
                    f"{np.mean(valid_ndwi):.3f}"
                    if valid_ndwi.size
                    else "N/A"
                ),
            )

        with metric3:

            metric_card(
                "Mean NDBI",
                (
                    f"{np.mean(valid_ndbi):.3f}"
                    if valid_ndbi.size
                    else "N/A"
                ),
            )


        # ----------------------------------------------------
        # LAND COVER
        # ----------------------------------------------------

        render_section(
            "Land Cover",
            description=(
                "Rule-based multispectral baseline"
            ),
            icon="🗺️",
        )

        status_badge(
            "RULE-BASED BASELINE — NOT A TRAINED ML MODEL",
            status="warning",
        )

        try:

            classification = classify_land_cover(
                ndvi=ndvi,
                ndwi=ndwi,
                ndbi=ndbi,
            )

            land_cover_figure = (
                create_land_cover_figure(
                    classification
                )
            )

            st.pyplot(
                land_cover_figure,
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "❌ Land-cover classification failed."
            )

            st.exception(error)

            classification = None


        if classification is not None:

            try:

                percentages = (
                    calculate_class_percentages(
                        classification
                    )
                )

                pc1, pc2, pc3, pc4, pc5 = (
                    st.columns(5)
                )

                with pc1:
                    st.metric(
                        "Vegetation",
                        f"{percentages['Vegetation']:.1f}%",
                    )

                with pc2:
                    st.metric(
                        "Water",
                        f"{percentages['Water']:.1f}%",
                    )

                with pc3:
                    st.metric(
                        "Built-up",
                        f"{percentages['Built-up']:.1f}%",
                    )

                with pc4:
                    st.metric(
                        "Bare Soil",
                        f"{percentages['Bare Soil']:.1f}%",
                    )

                with pc5:
                    st.metric(
                        "Other",
                        f"{percentages['Other']:.1f}%",
                    )

            except Exception:

                pass


            try:

                area = calculate_area_km2(
                    classification,
                    pixel_size_meters=10.0,
                )

                ac1, ac2, ac3, ac4 = (
                    st.columns(4)
                )

                with ac1:
                    st.metric(
                        "Vegetation",
                        f"{area['Vegetation']:.3f} km²",
                    )

                with ac2:
                    st.metric(
                        "Water",
                        f"{area['Water']:.3f} km²",
                    )

                with ac3:
                    st.metric(
                        "Built-up",
                        f"{area['Built-up']:.3f} km²",
                    )

                with ac4:
                    st.metric(
                        "Bare Soil",
                        f"{area['Bare Soil']:.3f} km²",
                    )

            except Exception:

                pass


        # ----------------------------------------------------
        # INDEX MAP
        # ----------------------------------------------------

        render_section(
            "Spectral Index Map",
            description=(
                "Pixel-level spectral interpretation"
            ),
            icon="🔬",
        )

        selected_index = st.selectbox(
            "Choose index",
            [
                "NDVI — Vegetation",
                "NDWI — Water",
                "NDBI — Built-up",
            ],
            key="main_index",
        )

        if selected_index.startswith("NDVI"):

            index_data = ndvi
            index_title = "NDVI — Vegetation"
            index_cmap = "RdYlGn"

        elif selected_index.startswith("NDWI"):

            index_data = ndwi
            index_title = "NDWI — Water"
            index_cmap = "Blues"

        else:

            index_data = ndbi
            index_title = "NDBI — Built-up"
            index_cmap = "Oranges"

        try:

            index_figure = create_index_figure(
                index_data,
                index_title,
                cmap=index_cmap,
            )

            st.pyplot(
                index_figure,
                use_container_width=True,
            )

        except Exception:

            st.warning(
                "⚠️ Could not render spectral map."
            )


# ============================================================
# CHANGE DETECTION TAB
# ============================================================

with change_tab:

    render_section(
        "Temporal Change Analysis",
        description=(
            "Compare two Sentinel-2 observations "
            "of the same area"
        ),
        icon="🔬",
    )

    if len(items) < 2:

        st.info(
            "Search for at least two Sentinel-2 scenes "
            "to activate Change Detection."
        )

    else:

        scene_options = {}

        for item in items:

            scene_date = (
                item.datetime.date()
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
                f"{scene_date} • "
                f"{cloud:.2f}% clouds • "
                f"{item.id}"
            )

            scene_options[label] = item

        scene_names = list(
            scene_options.keys()
        )

        change_col1, change_col2 = (
            st.columns(2)
        )

        with change_col1:

            before_name = st.selectbox(
                "📅 Data A — Before",
                scene_names,
                key="change_before",
            )

        with change_col2:

            after_name = st.selectbox(
                "📅 Data B — After",
                scene_names,
                index=min(
                    1,
                    len(scene_names) - 1,
                ),
                key="change_after",
            )

        settings_col1, settings_col2 = (
            st.columns(2)
        )

        with settings_col1:

            threshold = st.slider(
                "🎚️ Change sensitivity",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                key="change_threshold",
            )

        with settings_col2:

            change_index_choice = st.selectbox(
                "🔬 Index to compare",
                [
                    "NDVI — Vegetation",
                    "NDWI — Water",
                    "NDBI — Built-up",
                ],
                key="change_index",
            )

        if st.button(
            "🔍 Analyze Changes",
            type="primary",
            use_container_width=True,
            key="run_change_detection",
        ):

            before_item = scene_options[
                before_name
            ]

            after_item = scene_options[
                after_name
            ]

            if before_item.id == after_item.id:

                st.warning(
                    "⚠️ Choose two different scenes."
                )

            else:

                if (
                    before_item.datetime
                    and after_item.datetime
                    and before_item.datetime
                    > after_item.datetime
                ):

                    st.warning(
                        "⚠️ Data A is chronologically AFTER "
                        "Data B. The result will represent "
                        "the reverse time direction."
                    )

                bbox = (
                    drawn_aoi["bbox"]
                    if drawn_aoi
                    else create_bbox(
                        latitude=latitude,
                        longitude=longitude,
                        area_size=area_size,
                    )
                )

                try:

                    with st.spinner(
                        "🛰️ Downloading Data A..."
                    ):

                        before_bands = (
                            download_required_bands(
                                item=before_item,
                                bbox=bbox,
                                output_directory=(
                                    RAW_DIR / before_item.id
                                ),
                            )
                        )

                    with st.spinner(
                        "🛰️ Downloading Data B..."
                    ):

                        after_bands = (
                            download_required_bands(
                                item=after_item,
                                bbox=bbox,
                                output_directory=(
                                    RAW_DIR / after_item.id
                                ),
                            )
                        )

                    before_b04, before_m04 = read_band(
                        before_bands["B04"]
                    )

                    before_b03, before_m03 = read_band(
                        before_bands["B03"]
                    )

                    before_b08, before_m08 = read_band(
                        before_bands["B08"]
                    )

                    before_b11, before_m11 = read_band(
                        before_bands["B11"]
                    )

                    after_b04, after_m04 = read_band(
                        after_bands["B04"]
                    )

                    after_b03, after_m03 = read_band(
                        after_bands["B03"]
                    )

                    after_b08, after_m08 = read_band(
                        after_bands["B08"]
                    )

                    after_b11, after_m11 = read_band(
                        after_bands["B11"]
                    )

                    before_b03 = (
                        align_band_to_reference(
                            before_b03,
                            before_m03,
                            before_b04,
                            before_m04,
                        )
                    )

                    before_b08 = (
                        align_band_to_reference(
                            before_b08,
                            before_m08,
                            before_b04,
                            before_m04,
                        )
                    )

                    before_b11 = (
                        align_band_to_reference(
                            before_b11,
                            before_m11,
                            before_b04,
                            before_m04,
                        )
                    )

                    after_b03 = (
                        align_band_to_reference(
                            after_b03,
                            after_m03,
                            after_b04,
                            after_m04,
                        )
                    )

                    after_b08 = (
                        align_band_to_reference(
                            after_b08,
                            after_m08,
                            after_b04,
                            after_m04,
                        )
                    )

                    after_b11 = (
                        align_band_to_reference(
                            after_b11,
                            after_m11,
                            after_b04,
                            after_m04,
                        )
                    )

                    if change_index_choice.startswith(
                        "NDVI"
                    ):

                        before_index = calculate_ndvi(
                            before_b04,
                            before_b08,
                        )

                        after_index = calculate_ndvi(
                            after_b04,
                            after_b08,
                        )

                        index_name = (
                            "NDVI — Vegetation"
                        )

                    elif change_index_choice.startswith(
                        "NDWI"
                    ):

                        before_index = calculate_ndwi(
                            before_b03,
                            before_b08,
                        )

                        after_index = calculate_ndwi(
                            after_b03,
                            after_b08,
                        )

                        index_name = (
                            "NDWI — Water"
                        )

                    else:

                        before_index = calculate_ndbi(
                            before_b08,
                            before_b11,
                        )

                        after_index = calculate_ndbi(
                            after_b08,
                            after_b11,
                        )

                        index_name = (
                            "NDBI — Built-up"
                        )

                    difference = (
                        calculate_difference(
                            before_index,
                            after_index,
                            before_metadata=before_m04,
                            after_metadata=after_m04,
                        )
                    )

                    change_map = detect_change(
                        difference,
                        threshold=threshold,
                    )

                    statistics = (
                        calculate_change_statistics(
                            change_map,
                            pixel_size_meters=10.0,
                        )
                    )

                    st.session_state.change_result = {
                        "difference": difference,
                        "change_map": change_map,
                        "statistics": statistics,
                        "index_name": index_name,
                        "before_id": before_item.id,
                        "after_id": after_item.id,
                    }

                    st.success(
                        "✅ Change detection completed."
                    )

                except RasterValidationError as error:

                    st.error(
                        "❌ CRS mismatch — Data A and Data B "
                        "cannot be compared."
                    )

                    st.caption(str(error))

                except Exception as error:

                    st.error(
                        "❌ Change detection failed."
                    )

                    st.exception(error)


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    change_result = (
        st.session_state.change_result
    )

    if change_result:

        render_section(
            "Change Detection Result",
            description=change_result["index_name"],
            icon="📊",
        )

        statistics = (
            change_result["statistics"]
        )

        rc1, rc2, rc3 = st.columns(3)

        with rc1:

            metric_card(
                "Decrease",
                f"{statistics['decrease_km2']:.3f} km²",
            )

        with rc2:

            metric_card(
                "Increase",
                f"{statistics['increase_km2']:.3f} km²",
            )

        with rc3:

            metric_card(
                "Total Changed",
                f"{statistics['total_changed_km2']:.3f} km²",
            )

        try:

            change_figure = create_change_figure(
                change_result["change_map"],
                title=(
                    f"{change_result['index_name']} "
                    "Change Detection"
                ),
            )

            st.pyplot(
                change_figure,
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                "⚠️ Could not render change map."
            )

            st.caption(str(error))


# ============================================================
# GEOSPATIAL AI TAB
# ============================================================

with ai_tab:

    render_section(
        "Geospatial AI Engine",
        description=(
            "Remote-sensing computer vision pipeline "
            "for object detection"
        ),
        icon="🎯",
    )

    data = st.session_state.satellite_data

    if data is None:

        st.info(
            "Download a satellite scene from the Overview "
            "tab to activate Geospatial AI."
        )

    else:

        # ----------------------------------------------------
        # LOAD AI RGB IF NECESSARY
        # ----------------------------------------------------

        try:

            b02_ai, m02_ai = read_band(
                data["bands"]["B02"]
            )

            b03_ai, m03_ai = read_band(
                data["bands"]["B03"]
            )

            b04_ai, m04_ai = read_band(
                data["bands"]["B04"]
            )

            b03_ai = align_band_to_reference(
                b03_ai,
                m03_ai,
                b04_ai,
                m04_ai,
            )

            b02_ai = align_band_to_reference(
                b02_ai,
                m02_ai,
                b04_ai,
                m04_ai,
            )

            detection_rgb = normalize_rgb(
                red=b04_ai,
                green=b03_ai,
                blue=b02_ai,
            )

            validate_detection_image(
                detection_rgb
            )

        except Exception:

            detection_rgb = None


        if detection_rgb is None:

            st.warning(
                "⚠️ RGB image is unavailable for AI."
            )

        else:

            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------

            render_section(
                "Detection Input",
                description=(
                    "Sentinel-2 RGB prepared for inference"
                ),
                icon="🛰️",
            )

            st.image(
                detection_rgb,
                caption=(
                    "Sentinel-2 RGB • AI inference input"
                ),
                use_container_width=True,
            )


            # ------------------------------------------------
            # CONFIGURATION
            # ------------------------------------------------

            render_section(
                "Inference Configuration",
                description=(
                    "Tiling, confidence and class selection"
                ),
                icon="⚙️",
            )

            tc1, tc2 = st.columns(2)

            with tc1:

                tile_size = st.selectbox(
                    "Tile size",
                    [256, 512, 768, 1024],
                    index=1,
                    key="tile_size",
                )

            with tc2:

                tile_overlap = st.slider(
                    "Tile overlap",
                    min_value=0,
                    max_value=256,
                    value=64,
                    step=16,
                    key="tile_overlap",
                )

            if tile_overlap >= tile_size:

                st.error(
                    "❌ Overlap must be smaller than tile size."
                )

            else:

                try:

                    number_of_tiles = tile_count(
                        detection_rgb,
                        tile_size=tile_size,
                        overlap=tile_overlap,
                    )

                    metric_card(
                        "Image Tiles",
                        str(number_of_tiles),
                        "Generated inference windows",
                    )

                except Exception as error:

                    st.error(
                        "❌ Failed to calculate tiles."
                    )

                    st.exception(error)


            detection_threshold = st.slider(
                "Confidence threshold",
                min_value=0.10,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="object_confidence",
            )


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            render_section(
                "Geospatial AI Model",
                description=(
                    "Model registry and checkpoint status"
                ),
                icon="🧠",
            )

            try:

                model_ids = list_models()

                selected_model_id = st.selectbox(
                    "Model",
                    model_ids,
                    key="selected_model",
                )

                selected_model = get_model(
                    selected_model_id
                )

                detector = SatelliteDetector(
                    model_id=selected_model_id,
                    device="cpu",
                )

                model_info = detector.info()

                mc1, mc2, mc3 = st.columns(3)

                with mc1:

                    metric_card(
                        "Model",
                        model_info["model"],
                    )

                with mc2:

                    metric_card(
                        "Input Size",
                        (
                            f"{model_info['input_size']}×"
                            f"{model_info['input_size']}"
                        ),
                    )

                with mc3:

                    if model_info[
                        "checkpoint_available"
                    ]:

                        status_badge(
                            "MODEL READY",
                            status="online",
                        )

                    else:

                        status_badge(
                            "CHECKPOINT NOT INSTALLED",
                            status="warning",
                        )

                st.caption(
                    model_info["description"]
                )

            except Exception as error:

                detector = None
                model_info = None

                st.error(
                    "❌ Model registry failed."
                )

                st.exception(error)


            # ------------------------------------------------
            # CLASSES
            # ------------------------------------------------

            if detector is not None:

                detection_classes = st.multiselect(
                    "Classes of interest",
                    list(
                        model_info["classes"]
                    ),
                    default=list(
                        model_info["classes"][:2]
                    ),
                    key="object_classes",
                )


                with st.expander(
                    "ℹ️ View AI Architecture"
                ):

                    st.code(
                        """
Sentinel-2
    ↓
B02 + B03 + B04
    ↓
RGB normalization
    ↓
Image tiling
    ↓
Model Registry
    ↓
Remote Sensing Model
    ↓
Confidence filtering
    ↓
Object detections
    ↓
Bounding boxes
    ↓
Geospatial coordinates
                        """,
                        language="text",
                    )


                # --------------------------------------------
                # RUN
                # --------------------------------------------

                if st.button(
                    "🤖 Run Geospatial AI",
                    type="primary",
                    use_container_width=True,
                    key="run_geospatial_ai",
                ):

                    if not detection_classes:

                        st.warning(
                            "⚠️ Select at least one class."
                        )

                    elif tile_overlap >= tile_size:

                        st.error(
                            "⚠️ Invalid tile configuration."
                        )

                    elif not model_info[
                        "checkpoint_available"
                    ]:

                        st.info(
                            "🧠 Model pipeline is ready, "
                            "but the selected checkpoint "
                            "is not installed yet."
                        )

                        st.caption(
                            "No artificial detections "
                            "are generated."
                        )

                    else:

                        try:

                            with st.spinner(
                                "🧠 Running Geospatial AI..."
                            ):

                                tiles = create_tiles(
                                    detection_rgb,
                                    tile_size=tile_size,
                                    overlap=tile_overlap,
                                )

                                detections = (
                                    detector.predict_tiles(
                                        tiles,
                                        confidence=(
                                            detection_threshold
                                        ),
                                    )
                                )

                                detections = (
                                    filter_detections(
                                        detections,
                                        detection_threshold,
                                    )
                                )

                                detections = (
                                    filter_classes(
                                        detections,
                                        detection_classes,
                                    )
                                )

                                st.session_state.object_detections = (
                                    detections
                                )

                            st.success(
                                f"✅ {len(tiles)} tiles processed."
                            )

                        except Exception as error:

                            st.error(
                                "❌ AI inference failed."
                            )

                            st.exception(error)


                # --------------------------------------------
                # RESULTS
                # --------------------------------------------

                detections = (
                    st.session_state.object_detections
                )

                if detections:

                    render_section(
                        "Detection Results",
                        description=(
                            "Objects detected by the "
                            "selected remote-sensing model"
                        ),
                        icon="📊",
                    )

                    try:

                        summary = detection_summary(
                            detections
                        )

                        r1, r2 = st.columns(2)

                        with r1:

                            metric_card(
                                "Objects",
                                str(len(detections)),
                            )

                        with r2:

                            metric_card(
                                "Classes",
                                str(len(summary)),
                            )

                        detection_figure = (
                            draw_detections(
                                detection_rgb,
                                detections,
                            )
                        )

                        st.pyplot(
                            detection_figure,
                            use_container_width=True,
                        )

                        st.subheader(
                            "Detected Classes"
                        )

                        class_cols = st.columns(
                            max(
                                1,
                                min(
                                    len(summary),
                                    4,
                                ),
                            )
                        )

                        for index, (
                            label,
                            quantity,
                        ) in enumerate(
                            summary.items()
                        ):

                            with class_cols[
                                index % len(class_cols)
                            ]:

                                metric_card(
                                    label,
                                    str(quantity),
                                )


                        # ------------------------------------
                        # GEOREFERENCE / EXPORT
                        # ------------------------------------

                        render_section(
                            "Geospatial Export",
                            description=(
                                "Convert detections into "
                                "georeferenced GeoJSON"
                            ),
                            icon="📤",
                        )

                        try:

                            detections_gdf = (
                                georeference_detections(
                                    detections,
                                    transform=m04_ai[
                                        "transform"
                                    ],
                                    crs=str(
                                        m04_ai["crs"]
                                    ),
                                )
                            )

                            geojson_bytes = (
                                to_geojson_bytes(
                                    detections_gdf
                                )
                            )

                            st.download_button(
                                "⬇️ Export Detections (GeoJSON)",
                                data=geojson_bytes,
                                file_name=(
                                    f"{data['scene_id']}"
                                    "_detections.geojson"
                                ),
                                mime="application/geo+json",
                                use_container_width=True,
                            )

                            st.caption(
                                f"{len(detections_gdf)} "
                                "georeferenced detections, "
                                "WGS84 (EPSG:4326)."
                            )

                        except Exception as error:

                            st.warning(
                                "⚠️ Detections could not be "
                                "georeferenced for export."
                            )

                            st.caption(str(error))

                    except Exception as error:

                        st.error(
                            "❌ Could not render detections."
                        )

                        st.exception(error)

                else:

                    st.info(
                        "🔜 No detections available. "
                        "Connect a trained checkpoint "
                        "to activate inference."
                    )


# ============================================================
# PROCESSING PIPELINE
# ============================================================

render_section(
    "Processing Pipeline",
    description=(
        "Live state of the geospatial intelligence workflow"
    ),
    icon="🚀",
)

render_pipeline_status(
    scene_available=bool(items),
    imagery_available=data is not None,
    spectral_available=data is not None,
    change_available=(
        st.session_state.change_result is not None
    ),
    ai_available=bool(
        st.session_state.object_detections
    ),
)


# ============================================================
# FOOTER
# ============================================================

render_footer()

st.caption(
    "Spectral values are analytical measurements and "
    "should be interpreted according to sensor "
    "characteristics, spatial resolution and preprocessing."
)