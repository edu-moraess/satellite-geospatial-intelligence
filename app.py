import streamlit as st
from datetime import date

from src.catalog import (
    search_sentinel,
    create_bbox,
)

from src.downloader import (
    download_required_bands,
)

from src.config import RAW_DIR

from src.visualization import (
    create_rgb,
    create_false_color,
)


st.set_page_config(
    page_title="Satellite Geospatial Intelligence",
    page_icon="🛰️",
    layout="wide",
)


st.title(
    "🛰️ Satellite Geospatial Intelligence"
)

st.caption(
    "AI-powered Earth Observation Laboratory"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🗺️ Area of Interest"
)


latitude = st.sidebar.number_input(
    "Latitude",
    value=-23.5505,
    format="%.6f",
)


longitude = st.sidebar.number_input(
    "Longitude",
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


st.sidebar.header(
    "📅 Time Range"
)


start_date = st.sidebar.date_input(
    "Start date",
    value=date(2026, 1, 1),
)


end_date = st.sidebar.date_input(
    "End date",
    value=date(2026, 8, 23),
)


st.sidebar.header(
    "☁️ Cloud Filter"
)


max_cloud_cover = st.sidebar.slider(
    "Maximum cloud coverage (%)",
    min_value=0,
    max_value=100,
    value=10,
)


search_button = st.sidebar.button(
    "🔎 Search Satellite Data",
    type="primary",
)


# ============================================================
# SEARCH
# ============================================================

if search_button:

    if start_date > end_date:

        st.error(
            "Start date must be before end date."
        )

        st.stop()

    with st.spinner(
        "Searching Sentinel-2..."
    ):

        items = search_sentinel(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
            start_date=str(start_date),
            end_date=str(end_date),
            max_cloud_cover=max_cloud_cover,
        )

    if not items:

        st.warning(
            "No satellite images found "
            "with the selected parameters."
        )

        st.stop()

    st.success(
        f"{len(items)} satellite scenes found."
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader(
        "🛰️ Available Satellite Scenes"
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
        )

        with st.expander(
            f"{acquisition_date} — "
            f"{cloud:.2f}% clouds"
        ):

            st.write(
                f"Scene ID: `{item.id}`"
            )

            st.write(
                f"Cloud coverage: `{cloud:.2f}%`"
            )


            download_button = st.button(
                "⬇️ Download this scene",
                key=f"download_{index}",
            )


            if download_button:

                bbox = create_bbox(
                    latitude,
                    longitude,
                    area_size,
                )

                output_directory = (
                    RAW_DIR / item.id
                )

                with st.spinner(
                    "Downloading satellite bands..."
                ):

                    downloaded = (
                        download_required_bands(
                            item=item,
                            bbox=bbox,
                            output_directory=(
                                output_directory
                            ),
                        )
                    )

                st.success(
                    "Satellite data downloaded."
                )


                # ==========================================
                # RGB
                # ==========================================

                rgb = create_rgb(
                    downloaded["B02"],
                    downloaded["B03"],
                    downloaded["B04"],
                )


                # ==========================================
                # FALSE COLOR
                # ==========================================

                false_color = (
                    create_false_color(
                        downloaded["B03"],
                        downloaded["B04"],
                        downloaded["B08"],
                    )
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.subheader(
                        "🌍 RGB"
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


                st.info(
                    "B02/B03/B04/B08 downloaded "
                    "and processed successfully."
                )