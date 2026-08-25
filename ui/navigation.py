"""
ui/navigation.py — Optional Mission Control sidebar helper.
app.py currently inlines Mission Control; this module remains available
for alternate entry points without changing scientific behavior.
"""

from __future__ import annotations

from datetime import date

import streamlit as st


def render_mission_control_sidebar() -> dict:
    """Render AOI / temporal / filter controls. Returns control values."""
    st.markdown("**Mission Control**")
    st.caption("AOI · temporal window · scene filter")

    st.markdown("AOI")
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
        "Area size (deg)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        key="ui_area_size",
    )

    st.markdown("Temporal window")
    start_date = st.date_input("Start", value=date(2026, 1, 1), key="ui_start_date")
    end_date = st.date_input("End", value=date.today(), key="ui_end_date")

    st.markdown("Scene filter")
    max_cloud_cover = st.slider(
        "Max cloud cover",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        format="%d%%",
        key="ui_cloud_cover",
    )

    search_clicked = st.button(
        "Search Sentinel-2",
        type="primary",
        use_container_width=True,
        key="ui_search",
    )

    st.caption("Sentinel-2 · STAC · Earth Observation pipeline")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "area_size": area_size,
        "start_date": start_date,
        "end_date": end_date,
        "max_cloud_cover": max_cloud_cover,
        "search_clicked": search_clicked,
    }
