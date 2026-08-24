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

The selected satellite scene is stored in
Streamlit session state so the visualization
persists after interactions.
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
                start_date=str(start_date),
                end_date=str(end_date),
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


    # Store search results
    st.session_state.search_results = items


# ============================================================
# GET SEARCH RESULTS FROM SESSION
# ============================================================

items = st.session_state.get(
    "search_results",
    [],
)


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

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
                                output_directory=output_directory,
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
                # SAVE DOWNLOADED DATA IN SESSION
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
                    "✅ Download completed."
                )


                # Force Streamlit to rebuild
                # the interface using the saved data.
                st.rerun()


# ============================================================
# LOAD SELECTED SATELLITE DATA
# ============================================================

satellite_data = (
    st.session_state.satellite_data
)


if satellite_data:

    downloaded = (
        satellite_data["bands"]
    )


    # ========================================================
    # SCENE INFORMATION
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
            f"{satellite_data['cloud']:.2f}%",
        )


    with info3:

        st.metric(
            "Scene",
            satellite_data["scene_id"][
                :20
            ],
        )


    # ========================================================
    # CREATE RGB
    # ========================================================

    with st.spinner(
        "🎨 Processing RGB image..."
    ):

        try:

            rgb = create_rgb(
                downloaded["B02"],
                downloaded["B03"],
                downloaded["B04"],
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
    # CREATE FALSE COLOR
    # ========================================================

    with st.spinner(
        "🌱 Processing False Color image..."
    ):

        try:

            false_color = create_false_color(
                downloaded["B03"],
                downloaded["B04"],
                downloaded["B08"],
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
    # DISPLAY IMAGES
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
    # MULTISPECTRAL ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🔬 Multispectral Analysis"
    )


    # ========================================================
    # READ BANDS
    # ========================================================

    with st.spinner(
        "📡 Reading spectral bands..."
    ):

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
                "❌ Could not read downloaded bands."
            )

            st.exception(
                error
            )

            st.stop()


    # ========================================================
    # CALCULATE INDICES
    # ========================================================

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
    # VALID PIXELS
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

        st.metric(
            "🌱 Mean NDVI",
            (
                f"{np.mean(valid_ndvi):.3f}"
                if len(valid_ndvi)
                else "N/A"
            ),
        )


    with metric2:

        st.metric(
            "💧 Mean NDWI",
            (
                f"{np.mean(valid_ndwi):.3f}"
                if len(valid_ndwi)
                else "N/A"
            ),
        )


    with metric3:

        st.metric(
            "🏙️ Mean NDBI",
            (
                f"{np.mean(valid_ndbi):.3f}"
                if len(valid_ndbi)
                else "N/A"
            ),
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


    if index_selected.startswith(
        "NDVI"
    ):

        selected_index = ndvi

        title = (
            "NDVI — Vegetation"
        )

        colormap = "RdYlGn"

        description = (
            "NDVI highlights the "
            "spectral response associated "
            "with vegetation."
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
        "🧠 Generating spectral map..."
    ):

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


    # ========================================================
    # BANDS
    # ========================================================

    st.divider()

    st.subheader(
        "📡 Bands Downloaded"
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


    # ========================================================
    # STATUS
    # ========================================================

    st.success(
        "🚀 Satellite scene loaded successfully. "
        "RGB, False Color, NDVI, NDWI and NDBI "
        "are available for analysis."
    )