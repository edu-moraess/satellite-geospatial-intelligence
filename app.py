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

# ============================================================
# NEW PROFESSIONAL GEOSPATIAL MAP
# ============================================================

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
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "search_results": [],
    "satellite_data": None,
    "change_result": None,
    "object_detections": [],
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# GLOBAL DATA
# ============================================================

items = st.session_state.search_results
data = st.session_state.satellite_data

# IMPORTANT:
# Always initialize this before any conditional block.
# Prevents the previous NameError.
detection_rgb = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛰️ Satellite Geospatial Intelligence"
)

st.caption(
    "Earth Observation • Computer Vision • Geospatial AI"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📍 Area of Interest"
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


# ============================================================
# DATE RANGE
# ============================================================

st.sidebar.header(
    "📅 Satellite Date Range"
)

start_date = st.sidebar.date_input(
    "Start date",
    value=date(2026, 1, 1),
)

end_date = st.sidebar.date_input(
    "End date",
    value=date(2026, 8, 23),
)


# ============================================================
# CLOUD COVER
# ============================================================

st.sidebar.header(
    "☁️ Image Quality"
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
            "❌ Start date must be before the end date."
        )

        st.stop()

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
            )

            st.session_state.search_results = results

            # Clear previous analysis.
            st.session_state.satellite_data = None
            st.session_state.change_result = None
            st.session_state.object_detections = []

            st.rerun()

        except Exception as error:

            st.error(
                "❌ Satellite catalog search failed."
            )

            st.exception(error)
            st.stop()


# Refresh references after search.
items = st.session_state.search_results
data = st.session_state.satellite_data


# ============================================================
# SEARCH RESULTS
# ============================================================

if items:

    st.success(
        f"🛰️ {len(items)} satellite scenes found."
    )

    st.subheader(
        "Available Sentinel-2 Scenes"
    )

    for index, item in enumerate(items[:20]):

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

        with st.expander(
            f"{acquisition_date} • "
            f"{cloud:.2f}% clouds"
        ):

            st.write(
                f"**Scene ID:** `{item.id}`"
            )

            st.write(
                f"**Acquisition:** `{acquisition_date}`"
            )

            st.write(
                f"**Cloud coverage:** "
                f"`{cloud:.2f}%`"
            )

            if st.button(
                "⬇️ Download & Analyze",
                key=f"download_{index}",
                use_container_width=True,
            ):

                bbox = create_bbox(
                    latitude=latitude,
                    longitude=longitude,
                    area_size=area_size,
                )

                output_directory = (
                    RAW_DIR / item.id
                )

                with st.spinner(
                    "⬇️ Downloading satellite data..."
                ):

                    try:

                        downloaded = (
                            download_required_bands(
                                item=item,
                                bbox=bbox,
                                output_directory=(
                                    output_directory
                                ),
                            )
                        )

                    except Exception as error:

                        st.error(
                            "❌ Satellite download failed."
                        )

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

                st.success(
                    "✅ Satellite scene downloaded."
                )

                st.rerun()


# ============================================================
# SELECTED SCENE
# ============================================================

data = st.session_state.satellite_data

# Always initialize.
detection_rgb = None


if data:

    st.divider()

    st.header(
        "🛰️ Selected Satellite Scene"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Acquisition",
            data["date"],
        )

    with col2:

        st.metric(
            "Cloud Coverage",
            f"{data['cloud']:.2f}%",
        )

    with col3:

        st.metric(
            "Scene",
            data["scene_id"][:24],
        )


    # ========================================================
    # PROFESSIONAL GEOSPATIAL MAP
    # ========================================================

    st.divider()

    try:

        scene_bbox = create_bbox(
            latitude=data.get(
                "latitude",
                latitude,
            ),
            longitude=data.get(
                "longitude",
                longitude,
            ),
            area_size=data.get(
                "area_size",
                area_size,
            ),
        )

        render_map_panel(
            latitude=data.get(
                "latitude",
                latitude,
            ),
            longitude=data.get(
                "longitude",
                longitude,
            ),
            area_size=data.get(
                "area_size",
                area_size,
            ),
            bbox=scene_bbox,
            scene_id=data["scene_id"],
            acquisition_date=data["date"],
            cloud_cover=data["cloud"],
            key="main_geospatial_map",
        )

    except Exception as error:

        st.warning(
            "⚠️ Interactive geospatial map "
            "could not be rendered."
        )

        st.exception(error)


    # ========================================================
    # LOAD BANDS
    # ========================================================

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


    # ========================================================
    # ALIGN BANDS
    # ========================================================

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


    # ========================================================
    # RGB
    # ========================================================

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


    # ========================================================
    # AI RGB
    # ========================================================

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


    # ========================================================
    # FALSE COLOR
    # ========================================================

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


    # ========================================================
    # SATELLITE VISUALIZATION
    # ========================================================

    st.divider()

    st.header(
        "🌍 Satellite Visualization"
    )

    image1, image2 = st.columns(2)

    with image1:

        st.subheader(
            "🌍 Natural RGB"
        )

        st.image(
            rgb,
            caption="Sentinel-2 Natural Color",
            width="stretch",
        )

    with image2:

        st.subheader(
            "🌱 False Color"
        )

        st.image(
            false_color,
            caption="Sentinel-2 False Color",
            width="stretch",
        )


    # ========================================================
    # MULTISPECTRAL ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🔬 Multispectral Analysis"
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

        st.metric(
            "🌱 Mean NDVI",
            (
                f"{np.mean(valid_ndvi):.3f}"
                if valid_ndvi.size
                else "N/A"
            ),
        )

    with metric2:

        st.metric(
            "💧 Mean NDWI",
            (
                f"{np.mean(valid_ndwi):.3f}"
                if valid_ndwi.size
                else "N/A"
            ),
        )

    with metric3:

        st.metric(
            "🏙️ Mean NDBI",
            (
                f"{np.mean(valid_ndbi):.3f}"
                if valid_ndbi.size
                else "N/A"
            ),
        )


    # ========================================================
    # LAND COVER
    # ========================================================

    st.divider()

    st.header(
        "🗺️ Land Cover Classification"
    )

    st.caption(
        "Rule-based multispectral baseline "
        "using NDVI, NDWI and NDBI."
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


    # ========================================================
    # LAND COVER DISTRIBUTION
    # ========================================================

    if classification is not None:

        try:

            percentages = (
                calculate_class_percentages(
                    classification
                )
            )

            st.subheader(
                "📊 Land Cover Distribution"
            )

            pc1, pc2, pc3, pc4, pc5 = (
                st.columns(5)
            )

            with pc1:

                st.metric(
                    "🌳 Vegetation",
                    f"{percentages['Vegetation']:.1f}%",
                )

            with pc2:

                st.metric(
                    "💧 Water",
                    f"{percentages['Water']:.1f}%",
                )

            with pc3:

                st.metric(
                    "🏙️ Built-up",
                    f"{percentages['Built-up']:.1f}%",
                )

            with pc4:

                st.metric(
                    "🟫 Bare Soil",
                    f"{percentages['Bare Soil']:.1f}%",
                )

            with pc5:

                st.metric(
                    "⬜ Other",
                    f"{percentages['Other']:.1f}%",
                )

        except Exception:

            pass


        # ====================================================
        # AREA
        # ====================================================

        try:

            area = calculate_area_km2(
                classification,
                pixel_size_meters=10.0,
            )

            st.subheader(
                "📐 Estimated Area"
            )

            ac1, ac2, ac3, ac4 = (
                st.columns(4)
            )

            with ac1:

                st.metric(
                    "🌳 Vegetation",
                    f"{area['Vegetation']:.3f} km²",
                )

            with ac2:

                st.metric(
                    "💧 Water",
                    f"{area['Water']:.3f} km²",
                )

            with ac3:

                st.metric(
                    "🏙️ Built-up",
                    f"{area['Built-up']:.3f} km²",
                )

            with ac4:

                st.metric(
                    "🟫 Bare Soil",
                    f"{area['Bare Soil']:.3f} km²",
                )

        except Exception:

            pass


    # ========================================================
    # INDEX MAP
    # ========================================================

    st.divider()

    st.header(
        "🔬 Spectral Index Maps"
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

    if selected_index.startswith(
        "NDVI"
    ):

        index_data = ndvi
        index_title = "NDVI — Vegetation"
        index_cmap = "RdYlGn"

    elif selected_index.startswith(
        "NDWI"
    ):

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
# CHANGE DETECTION
# ============================================================

st.divider()

st.header(
    "🛰️ Change Detection"
)

st.caption(
    "Compare two Sentinel-2 observations "
    "of the same area to identify "
    "spectral changes over time."
)


if len(items) >= 2:

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


    threshold = st.slider(
        "🎚️ Change sensitivity",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        key="change_threshold",
    )


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

            bbox = create_bbox(
                latitude=latitude,
                longitude=longitude,
                area_size=area_size,
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


                before_b04, before_m04 = (
                    read_band(
                        before_bands["B04"]
                    )
                )

                before_b03, before_m03 = (
                    read_band(
                        before_bands["B03"]
                    )
                )

                before_b08, before_m08 = (
                    read_band(
                        before_bands["B08"]
                    )
                )

                before_b11, before_m11 = (
                    read_band(
                        before_bands["B11"]
                    )
                )


                after_b04, after_m04 = (
                    read_band(
                        after_bands["B04"]
                    )
                )

                after_b03, after_m03 = (
                    read_band(
                        after_bands["B03"]
                    )
                )

                after_b08, after_m08 = (
                    read_band(
                        after_bands["B08"]
                    )
                )

                after_b11, after_m11 = (
                    read_band(
                        after_bands["B11"]
                    )
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

            except Exception as error:

                st.error(
                    "❌ Change detection failed."
                )

                st.exception(error)

else:

    st.info(
        "ℹ️ Search for at least two satellite "
        "scenes to activate Change Detection."
    )


# ============================================================
# CHANGE RESULTS
# ============================================================

change_result = (
    st.session_state.change_result
)


if change_result:

    st.subheader(
        f"📊 {change_result['index_name']}"
    )

    statistics = (
        change_result["statistics"]
    )

    rc1, rc2, rc3 = st.columns(3)

    with rc1:

        st.metric(
            "🔴 Decrease",
            f"{statistics['decrease_km2']:.3f} km²",
        )

    with rc2:

        st.metric(
            "🟢 Increase",
            f"{statistics['increase_km2']:.3f} km²",
        )

    with rc3:

        st.metric(
            "🛰️ Total Changed",
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

    except Exception:

        pass


# ============================================================
# GEOSPATIAL AI
# ============================================================

st.divider()

st.header(
    "🎯 Geospatial AI"
)

st.caption(
    "Remote-sensing computer vision pipeline "
    "for object detection."
)


if data is None:

    st.info(
        "ℹ️ Download a satellite scene to "
        "activate Geospatial AI."
    )

elif detection_rgb is None:

    st.warning(
        "⚠️ RGB image is unavailable for AI."
    )

else:

    # ========================================================
    # INPUT
    # ========================================================

    st.subheader(
        "🛰️ Detection Input"
    )

    st.image(
        detection_rgb,
        caption=(
            "Sentinel-2 RGB prepared "
            "for Geospatial AI"
        ),
        width="stretch",
    )


    # ========================================================
    # TILING
    # ========================================================

    st.subheader(
        "🧩 AI Image Tiling"
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

            st.metric(
                "🧩 Image tiles",
                number_of_tiles,
            )

        except Exception as error:

            st.error(
                "❌ Failed to calculate tiles."
            )

            st.exception(error)


    # ========================================================
    # DETECTION CONFIGURATION
    # ========================================================

    st.subheader(
        "⚙️ Detection Configuration"
    )

    detection_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="object_confidence",
    )


    # ========================================================
    # MODEL REGISTRY
    # ========================================================

    st.subheader(
        "🧠 Geospatial AI Model"
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

            st.metric(
                "Model",
                model_info["model"],
            )

        with mc2:

            st.metric(
                "Input",
                (
                    f"{model_info['input_size']}×"
                    f"{model_info['input_size']}"
                ),
            )

        with mc3:

            if model_info[
                "checkpoint_available"
            ]:

                st.success(
                    "CHECKPOINT FOUND"
                )

            else:

                st.warning(
                    "CHECKPOINT MISSING"
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


    # ========================================================
    # CLASSES
    # ========================================================

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


        # ====================================================
        # ARCHITECTURE
        # ====================================================

        with st.expander(
            "ℹ️ AI Architecture"
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


        # ====================================================
        # RUN AI
        # ====================================================

        if st.button(
            "🤖 Run Geospatial AI",
            type="primary",
            use_container_width=True,
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


        # ====================================================
        # RESULTS
        # ====================================================

        detections = (
            st.session_state.object_detections
        )

        if detections:

            try:

                summary = detection_summary(
                    detections
                )

                st.subheader(
                    "📊 Detection Results"
                )

                r1, r2 = st.columns(2)

                with r1:

                    st.metric(
                        "Objects",
                        len(detections),
                    )

                with r2:

                    st.metric(
                        "Classes",
                        len(summary),
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
                    "🏷️ Detected Classes"
                )

                for label, quantity in (
                    summary.items()
                ):

                    st.write(
                        f"**{label}:** {quantity}"
                    )

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
# PIPELINE STATUS
# ============================================================

st.divider()

st.subheader(
    "🚀 Project Pipeline"
)

p1, p2, p3, p4, p5 = st.columns(5)

with p1:

    st.success(
        "✅ Scene Search"
    )

with p2:

    st.success(
        "✅ Spectral Analysis"
    )

with p3:

    st.success(
        "✅ Change Detection"
    )

with p4:

    st.success(
        "✅ Image Tiling"
    )

with p5:

    if data is not None:

        st.info(
            "🧠 AI Ready"
        )

    else:

        st.info(
            "🔄 AI Waiting"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Satellite Geospatial Intelligence • "
    "Earth Observation • Computer Vision • "
    "Geospatial AI"
)

st.caption(
    "Spectral values are analytical measurements "
    "and should be interpreted according to sensor "
    "characteristics, spatial resolution and "
    "preprocessing."
)