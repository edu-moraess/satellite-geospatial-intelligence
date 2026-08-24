"""
Satellite Geospatial Intelligence
==================================

Earth Observation
Computer Vision
Geospatial AI

Stage 2:
Land Cover Classification
"""

from datetime import date

import numpy as np
import streamlit as st


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


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# SESSION
# ============================================================

if "search_results" not in st.session_state:

    st.session_state.search_results = []


if "satellite_data" not in st.session_state:

    st.session_state.satellite_data = None


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
# DATES
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
# CLOUD
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
                start_date=str(
                    start_date
                ),
                end_date=str(
                    end_date
                ),
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

items = (
    st.session_state.search_results
)


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
# ANALYSIS
# ============================================================

data = (
    st.session_state.satellite_data
)


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
                "❌ Failed to load bands."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # ALIGN
    # ========================================================

    with st.spinner(
        "🔄 Aligning spectral grids..."
    ):

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


    # ========================================================
    # RGB
    # ========================================================

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
            use_container_width=True,
        )


    with image2:

        st.subheader(
            "🌱 False Color"
        )

        st.image(
            false_color,
            caption="Sentinel-2 False Color",
            use_container_width=True,
        )


    # ========================================================
    # SPECTRAL INDICES
    # ========================================================

    st.divider()

    st.header(
        "🔬 Multispectral Analysis"
    )


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


    # ========================================================
    # INDEX METRICS
    # ========================================================

    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        valid = ndvi[
            np.isfinite(ndvi)
        ]

        st.metric(
            "🌱 Mean NDVI",
            (
                f"{np.mean(valid):.3f}"
                if valid.size
                else "N/A"
            ),
        )


    with c2:

        valid = ndwi[
            np.isfinite(ndwi)
        ]

        st.metric(
            "💧 Mean NDWI",
            (
                f"{np.mean(valid):.3f}"
                if valid.size
                else "N/A"
            ),
        )


    with c3:

        valid = ndbi[
            np.isfinite(ndbi)
        ]

        st.metric(
            "🏙️ Mean NDBI",
            (
                f"{np.mean(valid):.3f}"
                if valid.size
                else "N/A"
            ),
        )


    # ========================================================
    # LAND COVER CLASSIFICATION
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

        classification = (
            classify_land_cover(
                ndvi=ndvi,
                ndwi=ndwi,
                ndbi=ndbi,
            )
        )


    # ========================================================
    # LAND COVER MAP
    # ========================================================

    figure = (
        create_land_cover_figure(
            classification
        )
    )


    st.pyplot(
        figure,
        use_container_width=True,
    )


    # ========================================================
    # PERCENTAGES
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
    # INDEX MAP
    # ========================================================

    st.divider()

    st.header(
        "🔬 Spectral Index Maps"
    )


    selected = st.selectbox(
        "Choose index",
        [
            "NDVI — Vegetation",
            "NDWI — Water",
            "NDBI — Built-up",
        ],
    )


    if selected.startswith(
        "NDVI"
    ):

        index_data = ndvi

        title = (
            "NDVI — Vegetation"
        )

        cmap = "RdYlGn"


    elif selected.startswith(
        "NDWI"
    ):

        index_data = ndwi

        title = (
            "NDWI — Water"
        )

        cmap = "Blues"


    else:

        index_data = ndbi

        title = (
            "NDBI — Built-up"
        )

        cmap = "Oranges"


    figure = create_index_figure(
        index_data,
        title,
        cmap=cmap,
    )


    st.pyplot(
        figure,
        use_container_width=True,
    )


    # ========================================================
    # STATUS
    # ========================================================

    st.divider()

    st.success(
        "🚀 Stage 2 completed: "
        "Land Cover Classification is active."
    )