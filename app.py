"""
Satellite Geospatial Intelligence
----------------------------------

Phase 1:
Satellite acquisition and visualization.

User controls:
- Latitude
- Longitude
- Area size
- Start date
- End date
- Cloud coverage
"""

from datetime import date

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

from src.visualization import (
    create_rgb,
    create_false_color,
)


# ============================================================
# PAGE CONFIG
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
# CLOUD FILTER
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
    # VALIDATE DATES
    # --------------------------------------------------------

    if start_date > end_date:

        st.error(
            "❌ Start date must be before "
            "the end date."
        )

        st.stop()

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

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
                max_cloud_cover=(
                    max_cloud_cover
                ),
            )

        except Exception as error:

            st.error(
                "Satellite catalog search failed."
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
            "No Sentinel-2 images were found "
            "for the selected parameters."
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
    # DISPLAY FIRST 10 RESULTS
    # --------------------------------------------------------

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

        title = (
            f"{acquisition_date}  •  "
            f"{cloud:.2f}% clouds"
        )

        with st.expander(
            title
        ):

            st.write(
                f"**Scene:** `{item.id}`"
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

                # ------------------------------------------------
                # CREATE AOI
                # ------------------------------------------------

                bbox = create_bbox(
                    latitude=latitude,
                    longitude=longitude,
                    area_size=area_size,
                )

                # ------------------------------------------------
                # OUTPUT DIRECTORY
                # ------------------------------------------------

                output_directory = (
                    RAW_DIR / item.id
                )

                output_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                with st.spinner(
                    "⬇️ Downloading satellite bands..."
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
                            "❌ Download failed."
                        )

                        st.exception(
                            error
                        )

                        st.stop()

                st.success(
                    "✅ Satellite data downloaded."
                )

                # ------------------------------------------------
                # RGB
                # ------------------------------------------------

                with st.spinner(
                    "🎨 Creating RGB composite..."
                ):

                    rgb = create_rgb(
                        downloaded["B02"],
                        downloaded["B03"],
                        downloaded["B04"],
                    )

                # ------------------------------------------------
                # FALSE COLOR
                # ------------------------------------------------

                with st.spinner(
                    "🌱 Creating false-color composite..."
                ):

                    false_color = (
                        create_false_color(
                            downloaded["B03"],
                            downloaded["B04"],
                            downloaded["B08"],
                        )
                    )

                # ------------------------------------------------
                # DISPLAY
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🛰️ Satellite Analysis"
                )

                col1, col2 = st.columns(
                    2
                )

                with col1:

                    st.markdown(
                        "### 🌍 Natural RGB"
                    )

                    st.image(
                        rgb,
                        use_container_width=True,
                    )

                with col2:

                    st.markdown(
                        "### 🌱 False Color"
                    )

                    st.image(
                        false_color,
                        use_container_width=True,
                    )

                # ------------------------------------------------
                # BAND INFORMATION
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "📡 Downloaded Bands"
                )

                band_col1, band_col2, band_col3, band_col4 = (
                    st.columns(4)
                )

                with band_col1:

                    st.metric(
                        "B02",
                        "Blue",
                    )

                with band_col2:

                    st.metric(
                        "B03",
                        "Green",
                    )

                with band_col3:

                    st.metric(
                        "B04",
                        "Red",
                    )

                with band_col4:

                    st.metric(
                        "B08",
                        "NIR",
                    )

                st.info(
                    "Phase 1 complete: "
                    "satellite acquisition, "
                    "AOI extraction and "
                    "basic visualization."
                )