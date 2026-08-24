"""
Satellite Geospatial Intelligence
==================================

Earth Observation
Computer Vision
Multispectral Analysis

Current capabilities:

- Sentinel-2 STAC search
- Date selection
- Cloud filtering
- AOI selection
- Satellite scene download
- RGB visualization
- False Color visualization
- NDVI
- NDWI
- NDBI
- Persistent Streamlit session state
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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "satellite_data" not in st.session_state:
    st.session_state.satellite_data = None

if "selected_scene" not in st.session_state:
    st.session_state.selected_scene = None


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
# SIDEBAR — AREA OF INTEREST
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


if search_button:

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
                max_cloud_cover=max_cloud_cover,
            )

        except Exception as error:

            st.error(
                "❌ Satellite catalog search failed."
            )

            st.exception(
                error
            )

            st.stop()


    st.session_state.search_results = items


# ============================================================
# SEARCH RESULTS
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

        cloud = item.properties.get(
            "eo:cloud_cover",
            0,
        )

        if item.datetime:

            acquisition_date = (
                item.datetime.date()
            )

        else:

            acquisition_date = (
                "Unknown"
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


            download_button = st.button(
                "⬇️ Download & Analyze",
                key=f"download_{index}",
                use_container_width=True,
            )


            if download_button:

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


                # ------------------------------------------------
                # SAVE TO SESSION
                # ------------------------------------------------

                st.session_state.satellite_data = {

                    "scene_id": item.id,

                    "date": str(
                        acquisition_date
                    ),

                    "cloud": float(
                        cloud
                    ),

                    "bands": downloaded,
                }


                st.session_state.selected_scene = (
                    item.id
                )


                st.success(
                    "✅ Satellite scene downloaded successfully."
                )


                st.rerun()


# ============================================================
# SATELLITE DATA
# ============================================================

satellite_data = (
    st.session_state.satellite_data
)


if satellite_data:

    downloaded = (
        satellite_data["bands"]
    )


    # ========================================================
    # SCENE HEADER
    # ========================================================

    st.divider()

    st.header(
        "🛰️ Selected Satellite Scene"
    )


    info1, info2, info3 = st.columns(
        3
    )


    with info1:

        st.metric(
            "Acquisition",
            satellite_data["date"],
        )


    with info2:

        st.metric(
            "Cloud Coverage",
            (
                f"{satellite_data['cloud']:.2f}%"
            ),
        )


    with info3:

        scene_id = (
            satellite_data["scene_id"]
        )

        st.metric(
            "Scene",
            scene_id[:20],
        )


    # ========================================================
    # READ VISUAL BANDS
    # ========================================================

    with st.spinner(
        "📡 Loading Sentinel-2 bands..."
    ):

        try:

            b02, meta_b02 = read_band(
                downloaded["B02"]
            )

            b03, meta_b03 = read_band(
                downloaded["B03"]
            )

            b04, meta_b04 = read_band(
                downloaded["B04"]
            )

            b08, meta_b08 = read_band(
                downloaded["B08"]
            )

            b11, meta_b11 = read_band(
                downloaded["B11"]
            )

        except Exception as error:

            st.error(
                "❌ Failed to read satellite bands."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # ALIGN BANDS
    # ========================================================

    with st.spinner(
        "🔄 Aligning spectral bands..."
    ):

        try:

            # ------------------------------------------------
            # B03 → B04
            # ------------------------------------------------

            b03_aligned = (
                align_band_to_reference(
                    band_array=b03,
                    band_metadata=meta_b03,
                    reference_array=b04,
                    reference_metadata=meta_b04,
                )
            )


            # ------------------------------------------------
            # B02 → B04
            # ------------------------------------------------

            b02_aligned = (
                align_band_to_reference(
                    band_array=b02,
                    band_metadata=meta_b02,
                    reference_array=b04,
                    reference_metadata=meta_b04,
                )
            )


            # ------------------------------------------------
            # B08 → B04
            # ------------------------------------------------

            b08_aligned = (
                align_band_to_reference(
                    band_array=b08,
                    band_metadata=meta_b08,
                    reference_array=b04,
                    reference_metadata=meta_b04,
                )
            )


            # ------------------------------------------------
            # B11 → B08
            #
            # This is the important fix.
            #
            # B11 can be 20 m while B08 is 10 m.
            # ------------------------------------------------

            b11_aligned = (
                align_band_to_reference(
                    band_array=b11,
                    band_metadata=meta_b11,
                    reference_array=b08,
                    reference_metadata=meta_b08,
                )
            )


        except Exception as error:

            st.error(
                "❌ Failed to align spectral bands."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # RGB
    # ========================================================

    with st.spinner(
        "🎨 Creating Natural RGB..."
    ):

        try:

            rgb = create_rgb(
                blue=b02_aligned,
                green=b03_aligned,
                red=b04,
            )

        except Exception as error:

            st.error(
                "❌ Failed to create RGB image."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # FALSE COLOR
    # ========================================================

    with st.spinner(
        "🌱 Creating False Color..."
    ):

        try:

            false_color = create_false_color(
                green=b03_aligned,
                red=b04,
                nir=b08_aligned,
            )

        except Exception as error:

            st.error(
                "❌ Failed to create False Color image."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.divider()

    st.header(
        "🌍 Satellite Visualization"
    )


    image_col1, image_col2 = st.columns(
        2
    )


    with image_col1:

        st.subheader(
            "🌍 Natural RGB"
        )

        st.image(
            rgb,
            caption="Sentinel-2 Natural Color",
            use_container_width=True,
        )


    with image_col2:

        st.subheader(
            "🌱 False Color"
        )

        st.image(
            false_color,
            caption="Sentinel-2 False Color",
            use_container_width=True,
        )


    # ========================================================
    # IMAGE QUALITY CHECK
    # ========================================================

    rgb_mean = float(
        np.mean(rgb)
    )

    false_color_mean = float(
        np.mean(false_color)
    )


    if rgb_mean == 0:

        st.warning(
            "⚠️ RGB image contains only zero values. "
            "Check the downloaded raster data."
        )


    if false_color_mean == 0:

        st.warning(
            "⚠️ False Color image contains only zero values. "
            "Check the downloaded raster data."
        )


    # ========================================================
    # MULTISPECTRAL ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🔬 Multispectral Analysis"
    )


    # ========================================================
    # CALCULATE INDICES
    # ========================================================

    with st.spinner(
        "🧠 Calculating spectral indices..."
    ):

        try:

            # ------------------------------------------------
            # NDVI
            # ------------------------------------------------

            ndvi = calculate_ndvi(
                red=b04,
                nir=b08_aligned,
            )


            # ------------------------------------------------
            # NDWI
            # ------------------------------------------------

            ndwi = calculate_ndwi(
                green=b03_aligned,
                nir=b08_aligned,
            )


            # ------------------------------------------------
            # NDBI
            # ------------------------------------------------

            ndbi = calculate_ndbi(
                nir=b08_aligned,
                swir=b11_aligned,
            )

        except Exception as error:

            st.error(
                "❌ Spectral index calculation failed."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # VALID VALUES
    # ========================================================

    valid_ndvi = ndvi[
        np.isfinite(ndvi)
    ]

    valid_ndwi = ndwi[
        np.isfinite(ndwi)
    ]

    valid_ndbi = ndbi[
        np.isfinite(ndbi)
    ]


    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3 = (
        st.columns(3)
    )


    with metric1:

        if valid_ndvi.size:

            ndvi_mean = np.mean(
                valid_ndvi
            )

            st.metric(
                "🌱 Mean NDVI",
                f"{ndvi_mean:.3f}",
            )

        else:

            st.metric(
                "🌱 Mean NDVI",
                "N/A",
            )


    with metric2:

        if valid_ndwi.size:

            ndwi_mean = np.mean(
                valid_ndwi
            )

            st.metric(
                "💧 Mean NDWI",
                f"{ndwi_mean:.3f}",
            )

        else:

            st.metric(
                "💧 Mean NDWI",
                "N/A",
            )


    with metric3:

        if valid_ndbi.size:

            ndbi_mean = np.mean(
                valid_ndbi
            )

            st.metric(
                "🏙️ Mean NDBI",
                f"{ndbi_mean:.3f}",
            )

        else:

            st.metric(
                "🏙️ Mean NDBI",
                "N/A",
            )


    # ========================================================
    # INDEX SELECTOR
    # ========================================================

    st.subheader(
        "🧠 Spectral Index"
    )


    index_selected = st.selectbox(
        "Choose an analysis:",
        [
            "NDVI — Vegetation",
            "NDWI — Water",
            "NDBI — Built-up",
        ],
        key="spectral_index",
    )


    # ========================================================
    # SELECT INDEX
    # ========================================================

    if index_selected.startswith(
        "NDVI"
    ):

        selected_index = ndvi

        title = (
            "NDVI — Vegetation"
        )

        colormap = "RdYlGn"

        description = (
            "NDVI highlights spectral "
            "responses associated with vegetation."
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
            "NDWI highlights spectral "
            "responses associated with water."
        )


    else:

        selected_index = ndbi

        title = (
            "NDBI — Built-up"
        )

        colormap = "Oranges"

        description = (
            "NDBI highlights spectral "
            "responses associated with "
            "built-up surfaces."
        )


    # ========================================================
    # INDEX MAP
    # ========================================================

    with st.spinner(
        "🗺️ Generating spectral map..."
    ):

        try:

            figure = create_index_figure(
                selected_index,
                title,
                cmap=colormap,
            )

        except Exception as error:

            st.error(
                "❌ Failed to generate spectral map."
            )

            st.exception(
                error
            )

            st.stop()


    st.pyplot(
        figure,
        use_container_width=True,
    )


    st.caption(
        description
    )


    # ========================================================
    # BAND INFORMATION
    # ========================================================

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
            "Blue • 10 m",
        )


    with band2:

        st.metric(
            "B03",
            "Green • 10 m",
        )


    with band3:

        st.metric(
            "B04",
            "Red • 10 m",
        )


    with band4:

        st.metric(
            "B08",
            "NIR • 10 m",
        )


    with band5:

        st.metric(
            "B11",
            "SWIR • aligned",
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    st.success(
        "🚀 Satellite scene loaded successfully. "
        "RGB, False Color, NDVI, NDWI and NDBI "
        "are available for analysis."
    )