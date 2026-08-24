from __future__ import annotations

from datetime import date

import streamlit as st


def render_sidebar() -> dict:
    """
    Render the application control panel.

    Returns:
        Dictionary containing AOI and search parameters.
    """

    with st.sidebar:

        st.markdown(
            """
            <div style="
                margin-bottom: 1.5rem;
            ">

                <div style="
                    font-size: 0.72rem;
                    color: #6F838D;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                ">
                    Geospatial Workspace
                </div>

                <div style="
                    font-size: 1.05rem;
                    font-weight: 700;
                    margin-top: 0.35rem;
                ">
                    Earth Observation
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Area of Interest"
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=-23.5505,
            format="%.6f",
            key="ui_latitude",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-46.6333,
            format="%.6f",
            key="ui_longitude",
        )

        area_size = st.slider(
            "Area size",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="ui_area_size",
        )

        st.divider()

        st.markdown(
            "### Temporal Query"
        )

        start_date = st.date_input(
            "Start date",
            value=date(2026, 1, 1),
            key="ui_start_date",
        )

        end_date = st.date_input(
            "End date",
            value=date.today(),
            key="ui_end_date",
        )

        st.divider()

        st.markdown(
            "### Image Quality"
        )

        max_cloud_cover = st.slider(
            "Maximum cloud coverage",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            format="%d%%",
            key="ui_cloud_cover",
        )

        st.divider()

        search_clicked = st.button(
            "Search Satellite Catalog",
            type="primary",
            use_container_width=True,
        )

        st.markdown(
            """
            <div style="
                margin-top: 1.5rem;
                color: #52656E;
                font-size: 0.67rem;
                line-height: 1.5;
            ">
                Sentinel-2 Earth Observation
                <br>
                STAC / Remote Sensing Pipeline
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "area_size": area_size,
        "start_date": start_date,
        "end_date": end_date,
        "max_cloud_cover": max_cloud_cover,
        "search_clicked": search_clicked,
    }