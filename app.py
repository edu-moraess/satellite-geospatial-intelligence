"""
Satellite Geospatial Intelligence
=================================

Phase 1
-------
Satellite acquisition
RGB
False Color

Phase 2
-------
NDVI
NDWI
NDBI

The user controls:
- Latitude
- Longitude
- AOI size
- Start date
- End date
- Cloud coverage
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=(
        "Satellite Geospatial Intelligence"
    ),
    page_icon="🛰️",
    layout="wide",
)


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
# SIDEBAR — AREA
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
# SIDEBAR — DATE
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
# SIDEBAR — CLOUD
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
# SEARCH BUTTON
# ============================================================

search_button = st.sidebar.button(
    "🔎 Search Satellite Data",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SEARCH
# ============================================================

if search_button:

    # --------------------------------------------------------
    # DATE VALIDATION
    # --------------------------------------------------------

    if start_date > end_date:

        st.error(
            "❌ Start date must be before "
            "the end date."
        )

        st.stop()

    # --------------------------------------------------------
    # SEARCH SATELLITE CATALOG
    # --------------------------------------------------------

    with st.spinner(
        "🛰️ Searching Sentinel-2..."
    ):

        try:

            items = search_sentinel(
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
            )

        except Exception as error:

            st.error(
                "❌ Satellite catalog search failed."
            )

            st.exception(
                error
            )

            st.stop()

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not items:

        st.warning(
            "No satellite scenes were found "
            "with the selected parameters."
        )

        st.stop()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.success(
        f"🛰️ {len(items)} satellite scenes found."
    )

    st.subheader(
        "Available Sentinel-2 Scenes"
    )

    # --------------------------------------------------------
    # DISPLAY SCENES
    # --------------------------------------------------------

    for index, item in enumerate(
        items[:10]
    ):

        cloud = item.properties.get(
            "eo:cloud_cover",
            0,
        )

        acquisition_date = (
            item.datetime.date()
            if item.datetime
            else "Unknown"
        )

        scene_title = (
            f"{acquisition_date} • "
            f"{cloud:.2f}% clouds"
        )

        with st.expander(
            scene_title
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

            # ------------------------------------------------
            # DOWNLOAD BUTTON
            # ------------------------------------------------

            download_button = st.button(
                "⬇️ Download & Analyze",
                key=(
                    f"download_{index}"
                ),
                use_container_width=True,
            )

            if download_button:

                # ============================================
                # CREATE AOI
                # ============================================

                bbox = create_bbox(
                    latitude=latitude,
                    longitude=longitude,
                    area_size=area_size,
                )

                # ============================================
                # OUTPUT DIRECTORY
                # ============================================

                output_directory = (
                    RAW_DIR / item.id
                )

                output_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ============================================
                # DOWNLOAD BANDS
                # ============================================

                with st.spinner(
                    "⬇️ Downloading Sentinel-2 bands..."
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

                st.success(
                    "✅ Satellite data downloaded."
                )

                # ============================================
                # RGB
                # ============================================

                with st.spinner(
                    "🎨 Creating RGB..."
                ):

                    rgb = create_rgb(
                        downloaded["B02"],
                        downloaded["B03"],
                        downloaded["B04"],
                    )

                # ============================================
                # FALSE COLOR
                # ============================================

                with st.spinner(
                    "🌱 Creating False Color..."
                ):

                    false_color = (
                        create_false_color(
                            downloaded["B03"],
                            downloaded["B04"],
                            downloaded["B08"],
                        )
                    )

                # ============================================
                # PHASE 1 VISUALIZATION
                # ============================================

                st.divider()

                st.header(
                    "🛰️ Satellite Visualization"
                )

                col1, col2 = st.columns(
                    2
                )

                with col1:

                    st.subheader(
                        "🌍 Natural RGB"
                    )

                    st.image(
                        rgb,
                        use_container_width=True,
                    )

                with col2:

                    st.subheader(
                        "🌱 False Color"
                    )

                    st.image(
                        false_color,
                        use_container_width=True,
                    )

                # ============================================
                # PHASE 2
                # MULTISPECTRAL ANALYSIS
                # ============================================

                st.divider()

                st.header(
                    "🔬 Multispectral Analysis"
                )

                st.caption(
                    "Spectral indices calculated "
                    "from Sentinel-2 bands."
                )

                # ============================================
                # READ BANDS
                # ============================================

                try:

                    b03, _ = read_band(
                        downloaded["B03"]
                    )

                    b04, _ = read_band(
                        downloaded["B04"]
                    )

                    b08, _ = read_band(
                        downloaded["B08"]
                    )

                    b11, _ = read_band(
                        downloaded["B11"]
                    )

                except Exception as error:

                    st.error(
                        "❌ Could not read "
                        "downloaded bands."
                    )

                    st.exception(
                        error
                    )

                    st.stop()

                # ============================================
                # CALCULATE INDICES
                # ============================================

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

                # ============================================
                # VALID PIXELS
                # ============================================

                valid_ndvi = ndvi[
                    np.isfinite(ndvi)
                ]

                valid_ndwi = ndwi[
                    np.isfinite(ndwi)
                ]

                valid_ndbi = ndbi[
                    np.isfinite(ndbi)
                ]

                # ============================================
                # METRICS
                # ============================================

                metric1, metric2, metric3 = (
                    st.columns(3)
                )

                with metric1:

                    if len(valid_ndvi):

                        st.metric(
                            "🌱 Mean NDVI",
                            f"{np.mean(valid_ndvi):.3f}",
                        )

                    else:

                        st.metric(
                            "🌱 Mean NDVI",
                            "N/A",
                        )

                with metric2:

                    if len(valid_ndwi):

                        st.metric(
                            "💧 Mean NDWI",
                            f"{np.mean(valid_ndwi):.3f}",
                        )

                    else:

                        st.metric(
                            "💧 Mean NDWI",
                            "N/A",
                        )

                with metric3:

                    if len(valid_ndbi):

                        st.metric(
                            "🏙️ Mean NDBI",
                            f"{np.mean(valid_ndbi):.3f}",
                        )

                    else:

                        st.metric(
                            "🏙️ Mean NDBI",
                            "N/A",
                        )

                # ============================================
                # INDEX SELECTOR
                # ============================================

                st.subheader(
                    "Spectral Index"
                )

                index_selected = (
                    st.selectbox(
                        "Select analysis",
                        [
                            "NDVI — Vegetation",
                            "NDWI — Water",
                            "NDBI — Built-up",
                        ],
                        key=(
                            f"index_{index}"
                        ),
                    )
                )

                # ============================================
                # SELECT INDEX
                # ============================================

                if index_selected.startswith(
                    "NDVI"
                ):

                    selected_index = ndvi

                    title = (
                        "NDVI — Vegetation"
                    )

                    colormap = "RdYlGn"

                    description = (
                        "NDVI highlights "
                        "vegetation vigor."
                    )

                elif index_selected.startswith(
                    "NDWI"
                ):

                    selected_index = ndwi

                    title = (
                        "NDWI — Water"
                    )

                    colormap = "Blues"

                    description = (
                        "NDWI highlights "
                        "water-related spectral response."
                    )

                else:

                    selected_index = ndbi

                    title = (
                        "NDBI — Built-up"
                    )

                    colormap = "Oranges"

                    description = (
                        "NDBI highlights "
                        "built-up spectral response."
                    )

                # ============================================
                # INDEX MAP
                # ============================================

                figure = create_index_figure(
                    selected_index,
                    title,
                    cmap=colormap,
                )

                st.pyplot(
                    figure,
                    use_container_width=True,
                )

                st.caption(
                    description
                )

                # ============================================
                # BAND INFORMATION
                # ============================================

                st.divider()

                st.subheader(
                    "📡 Bands Used"
                )

                band1, band2, band3, band4, band5 = (
                    st.columns(5)
                )

                with band1:

                    st.metric(
                        "B02",
                        "Blue",
                    )

                with band2:

                    st.metric(
                        "B03",
                        "Green",
                    )

                with band3:

                    st.metric(
                        "B04",
                        "Red",
                    )

                with band4:

                    st.metric(
                        "B08",
                        "NIR",
                    )

                with band5:

                    st.metric(
                        "B11",
                        "SWIR",
                    )

                # ============================================
                # STATUS
                # ============================================

                st.success(
                    "✅ Phase 2 complete: "
                    "NDVI, NDWI and NDBI "
                    "calculated successfully."
                )