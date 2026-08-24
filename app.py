"""
SATELLITE GEOSPATIAL INTELLIGENCE
=================================

Earth Observation
Computer Vision
Geospatial AI

Pipeline:

1. Sentinel-2 catalog search
2. Satellite scene selection
3. Satellite band download
4. RGB visualization
5. False Color visualization
6. NDVI / NDWI / NDBI
7. Land Cover Classification
8. Change Detection
9. Object Detection preparation

Next stage:
10. Real object detection model
11. Bounding boxes
12. Geospatial object statistics
13. Temporal object tracking
"""

from datetime import date

import numpy as np
import streamlit as st


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.catalog import (
    search_sentinel,
    create_bbox,
)

from src.config import (
    RAW_DIR,
)

from src.downloader import (
    download_required_bands,
)

from src.geospatial import (
    read_band,
    align_band_to_reference,
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
    create_difference_figure,
)

from src.object_detection import (
    normalize_rgb,
    validate_detection_image,
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

if "search_results" not in st.session_state:
    st.session_state.search_results = []


if "satellite_data" not in st.session_state:
    st.session_state.satellite_data = None


if "change_result" not in st.session_state:
    st.session_state.change_result = None


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
    value=date(
        2026,
        1,
        1,
    ),
)


end_date = st.sidebar.date_input(
    "End date",
    value=date(
        2026,
        8,
        23,
    ),
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
            "❌ Start date must be before "
            "the end date."
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

            st.session_state.search_results = (
                results
            )

        except Exception as error:

            st.error(
                "❌ Satellite catalog search failed."
            )

            st.exception(
                error
            )

            st.stop()


# ============================================================
# RESULTS
# ============================================================

items = st.session_state.search_results


if items:

    st.success(
        f"🛰️ {len(items)} satellite scenes found."
    )

    st.subheader(
        "Available Sentinel-2 Scenes"
    )


    for index, item in enumerate(
        items[:10]
    ):

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
                f"**Acquisition:** "
                f"`{acquisition_date}`"
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

                        st.exception(
                            error
                        )

                        st.stop()


                st.session_state.satellite_data = {

                    "scene_id": item.id,

                    "date": str(
                        acquisition_date
                    ),

                    "cloud": cloud,

                    "bands": downloaded,
                }


                st.success(
                    "✅ Satellite scene downloaded."
                )

                st.rerun()


# ============================================================
# SELECTED SCENE
# ============================================================

data = st.session_state.satellite_data


if data:

    st.divider()

    st.header(
        "🛰️ Selected Satellite Scene"
    )


    col1, col2, col3 = st.columns(
        3
    )


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
            data["scene_id"][:20],
        )


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

            st.exception(
                error
            )

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
                "❌ Failed to align satellite bands."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # RGB
    # ========================================================

    rgb = create_rgb(
        blue=b02,
        green=b03,
        red=b04,
    )


    # ========================================================
    # FALSE COLOR
    # ========================================================

    false_color = create_false_color(
        green=b03,
        red=b04,
        nir=b08,
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.divider()

    st.header(
        "🌍 Satellite Visualization"
    )


    image1, image2 = st.columns(
        2
    )


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
    # SPECTRAL INDICES
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

        st.exception(
            error
        )

        st.stop()


    # ========================================================
    # INDEX METRICS
    # ========================================================

    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        valid_ndvi = ndvi[
            np.isfinite(ndvi)
        ]

        st.metric(
            "🌱 Mean NDVI",
            (
                f"{np.mean(valid_ndvi):.3f}"
                if valid_ndvi.size
                else "N/A"
            ),
        )


    with c2:

        valid_ndwi = ndwi[
            np.isfinite(ndwi)
        ]

        st.metric(
            "💧 Mean NDWI",
            (
                f"{np.mean(valid_ndwi):.3f}"
                if valid_ndwi.size
                else "N/A"
            ),
        )


    with c3:

        valid_ndbi = ndbi[
            np.isfinite(ndbi)
        ]

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


    with st.spinner(
        "🧠 Classifying satellite pixels..."
    ):

        try:

            classification = (
                classify_land_cover(
                    ndvi=ndvi,
                    ndwi=ndwi,
                    ndbi=ndbi,
                )
            )

        except Exception as error:

            st.error(
                "❌ Land-cover classification failed."
            )

            st.exception(
                error
            )

            st.stop()


    land_cover_figure = (
        create_land_cover_figure(
            classification
        )
    )


    st.pyplot(
        land_cover_figure,
        use_container_width=True,
    )


    # ========================================================
    # LAND COVER PERCENTAGES
    # ========================================================

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


    # ========================================================
    # AREA
    # ========================================================

    st.subheader(
        "📐 Estimated Area"
    )


    area = calculate_area_km2(
        classification,
        pixel_size_meters=10.0,
    )


    area1, area2, area3, area4 = (
        st.columns(4)
    )


    with area1:

        st.metric(
            "🌳 Vegetation",
            f"{area['Vegetation']:.3f} km²",
        )


    with area2:

        st.metric(
            "💧 Water",
            f"{area['Water']:.3f} km²",
        )


    with area3:

        st.metric(
            "🏙️ Built-up",
            f"{area['Built-up']:.3f} km²",
        )


    with area4:

        st.metric(
            "🟫 Bare Soil",
            f"{area['Bare Soil']:.3f} km²",
        )


    # ========================================================
    # SPECTRAL INDEX MAP
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

        index_title = (
            "NDVI — Vegetation"
        )

        index_cmap = "RdYlGn"


    elif selected_index.startswith(
        "NDWI"
    ):

        index_data = ndwi

        index_title = (
            "NDWI — Water"
        )

        index_cmap = "Blues"


    else:

        index_data = ndbi

        index_title = (
            "NDBI — Built-up"
        )

        index_cmap = "Oranges"


    index_figure = create_index_figure(
        index_data,
        index_title,
        cmap=index_cmap,
    )


    st.pyplot(
        index_figure,
        use_container_width=True,
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


if len(items) < 2:

    st.info(
        "ℹ️ Search for at least two "
        "satellite scenes to activate "
        "Change Detection."
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


    col_a, col_b = st.columns(
        2
    )


    with col_a:

        st.subheader(
            "📅 Data A — Before"
        )

        before_name = st.selectbox(
            "Satellite scene A",
            scene_names,
            key="change_before",
        )


    with col_b:

        st.subheader(
            "📅 Data B — After"
        )

        after_name = st.selectbox(
            "Satellite scene B",
            scene_names,
            index=(
                1
                if len(scene_names) > 1
                else 0
            ),
            key="change_after",
        )


    threshold = st.slider(
        "🎚️ Change sensitivity",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
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

        before_item = (
            scene_options[
                before_name
            ]
        )

        after_item = (
            scene_options[
                after_name
            ]
        )


        if (
            before_item.id
            == after_item.id
        ):

            st.warning(
                "⚠️ Escolha duas cenas diferentes."
            )

            st.stop()


        bbox = create_bbox(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
        )


        # ----------------------------------------------------
        # DATA A
        # ----------------------------------------------------

        with st.spinner(
            "🛰️ Downloading Data A..."
        ):

            try:

                before_directory = (
                    RAW_DIR
                    / before_item.id
                )


                before_bands = (
                    download_required_bands(
                        item=before_item,
                        bbox=bbox,
                        output_directory=(
                            before_directory
                        ),
                    )
                )

            except Exception as error:

                st.error(
                    "❌ Failed to download Data A."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # DATA B
        # ----------------------------------------------------

        with st.spinner(
            "🛰️ Downloading Data B..."
        ):

            try:

                after_directory = (
                    RAW_DIR
                    / after_item.id
                )


                after_bands = (
                    download_required_bands(
                        item=after_item,
                        bbox=bbox,
                        output_directory=(
                            after_directory
                        ),
                    )
                )

            except Exception as error:

                st.error(
                    "❌ Failed to download Data B."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # LOAD DATA A
        # ----------------------------------------------------

        with st.spinner(
            "📡 Loading Data A..."
        ):

            try:

                before_b03, before_m03 = (
                    read_band(
                        before_bands["B03"]
                    )
                )

                before_b04, before_m04 = (
                    read_band(
                        before_bands["B04"]
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

            except Exception as error:

                st.error(
                    "❌ Failed to read Data A."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # LOAD DATA B
        # ----------------------------------------------------

        with st.spinner(
            "📡 Loading Data B..."
        ):

            try:

                after_b03, after_m03 = (
                    read_band(
                        after_bands["B03"]
                    )
                )

                after_b04, after_m04 = (
                    read_band(
                        after_bands["B04"]
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

            except Exception as error:

                st.error(
                    "❌ Failed to read Data B."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # ALIGN DATA A
        # ----------------------------------------------------

        with st.spinner(
            "🔄 Aligning Data A..."
        ):

            try:

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

            except Exception as error:

                st.error(
                    "❌ Failed to align Data A."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # ALIGN DATA B
        # ----------------------------------------------------

        with st.spinner(
            "🔄 Aligning Data B..."
        ):

            try:

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

            except Exception as error:

                st.error(
                    "❌ Failed to align Data B."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # CALCULATE INDEX
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Calculating spectral changes..."
        ):

            try:

                if change_index_choice.startswith(
                    "NDVI"
                ):

                    before_index = (
                        calculate_ndvi(
                            red=before_b04,
                            nir=before_b08,
                        )
                    )

                    after_index = (
                        calculate_ndvi(
                            red=after_b04,
                            nir=after_b08,
                        )
                    )

                    index_name = (
                        "NDVI — Vegetation"
                    )


                elif change_index_choice.startswith(
                    "NDWI"
                ):

                    before_index = (
                        calculate_ndwi(
                            green=before_b03,
                            nir=before_b08,
                        )
                    )

                    after_index = (
                        calculate_ndwi(
                            green=after_b03,
                            nir=after_b08,
                        )
                    )

                    index_name = (
                        "NDWI — Water"
                    )


                else:

                    before_index = (
                        calculate_ndbi(
                            nir=before_b08,
                            swir=before_b11,
                        )
                    )

                    after_index = (
                        calculate_ndbi(
                            nir=after_b08,
                            swir=after_b11,
                        )
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


                change_map = (
                    detect_change(
                        difference,
                        threshold=threshold,
                    )
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


            except Exception as error:

                st.error(
                    "❌ Change detection failed."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.success(
            "✅ Change detection completed."
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


    result1, result2, result3 = (
        st.columns(3)
    )


    with result1:

        st.metric(
            "🔴 Decrease",
            (
                f"{statistics['decrease_km2']:.3f} "
                "km²"
            ),
        )


    with result2:

        st.metric(
            "🟢 Increase",
            (
                f"{statistics['increase_km2']:.3f} "
                "km²"
            ),
        )


    with result3:

        st.metric(
            "🛰️ Total Changed",
            (
                f"{statistics['total_changed_km2']:.3f} "
                "km²"
            ),
        )


    # ========================================================
    # CHANGE MAP
    # ========================================================

    st.subheader(
        "🗺️ Change Map"
    )


    change_figure = (
        create_change_figure(
            change_result["change_map"],
            title=(
                f"{change_result['index_name']} "
                "Change Detection"
            ),
        )
    )


    st.pyplot(
        change_figure,
        use_container_width=True,
    )


    # ========================================================
    # CONTINUOUS DIFFERENCE
    # ========================================================

    with st.expander(
        "🔬 View continuous spectral difference"
    ):

        difference_figure = (
            create_difference_figure(
                change_result["difference"],
                title=(
                    f"{change_result['index_name']} "
                    "— Continuous Difference"
                ),
            )
        )


        st.pyplot(
            difference_figure,
            use_container_width=True,
        )


# ============================================================
# OBJECT DETECTION
# ============================================================

st.divider()

st.header(
    "🎯 Object Detection"
)

st.caption(
    "Computer Vision pipeline for preparing "
    "satellite imagery for object detection."
)


if data:

    st.subheader(
        "🛰️ Detection Input"
    )


    # ========================================================
    # PREPARE RGB
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


    except Exception as error:

        st.error(
            "❌ Could not prepare "
            "the detection image."
        )

        st.exception(
            error

        )

        detection_rgb = None


    # ========================================================
    # DISPLAY
    # ========================================================

    if detection_rgb is not None:

        st.image(
            detection_rgb,
            caption=(
                "Satellite RGB prepared "
                "for object detection"
            ),
            width="stretch",
        )


        # ====================================================
        # DETECTION SETTINGS
        # ====================================================

        st.subheader(
            "⚙️ Detection Configuration"
        )


        detection_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="detection_threshold",
        )


        detection_classes = st.multiselect(
            "Classes of interest",
            [
                "Buildings",
                "Roads",
                "Water",
                "Vegetation",
                "Vehicles",
            ],
            default=[
                "Buildings",
                "Roads",
                "Water",
            ],
            key="detection_classes",
        )


        # ====================================================
        # DETECTION STATUS
        # ====================================================

        st.info(
            "🧠 Image preparation completed. "
            "The real object-detection model "
            "will be integrated in the next "
            "sub-stage."
        )


        # ====================================================
        # TECHNICAL INFORMATION
        # ====================================================

        st.subheader(
            "🔬 Detection Pipeline"
        )


        pipeline1, pipeline2, pipeline3, pipeline4 = (
            st.columns(4)
        )


        with pipeline1:

            st.metric(
                "Input",
                "Sentinel-2 RGB",
            )


        with pipeline2:

            st.metric(
                "Channels",
                "3",
            )


        with pipeline3:

            st.metric(
                "Confidence",
                f"{detection_threshold:.0%}",
            )


        with pipeline4:

            st.metric(
                "Classes",
                len(detection_classes),
            )


        # ====================================================
        # SELECTED CLASSES
        # ====================================================

        if detection_classes:

            st.write(
                "**Selected classes:** "
                + ", ".join(
                    detection_classes
                )
            )

        else:

            st.warning(
                "⚠️ Select at least one "
                "class of interest."
            )


else:

    st.info(
        "ℹ️ Download a satellite scene "
        "above to activate Object Detection."
    )


# ============================================================
# PROJECT STATUS
# ============================================================

st.divider()

st.subheader(
    "🚀 Project Pipeline"
)


status1, status2, status3, status4, status5 = (
    st.columns(5)
)


with status1:

    st.success(
        "✅ Scene Search"
    )


with status2:

    st.success(
        "✅ Spectral Analysis"
    )


with status3:

    st.success(
        "✅ Change Detection"
    )


with status4:

    st.success(
        "✅ Detection Input"
    )


with status5:

    st.info(
        "🔜 AI Detection"
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
    "and should be interpreted according to "
    "sensor characteristics and preprocessing."
)