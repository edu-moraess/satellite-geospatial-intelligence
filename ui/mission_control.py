"""Mission Control UI: analysis inputs and mission summary."""
from __future__ import annotations
from datetime import date
import streamlit as st
from src.sensor_registry import SENSORS, get_sensor
from ui.layout import status_badge


DEFAULT_LATITUDE = -23.550520
DEFAULT_LONGITUDE = -46.633308


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div class="eyebrow">GEOCORE</div>'
            '<div class="sidebar-title">Mission Control</div>'
            '<div class="sidebar-subtitle">Sensor · AOI · temporal window</div>',
            unsafe_allow_html=True,
        )

        sensor_options = {sensor.name: sensor.id for sensor in SENSORS.values()}
        selected_name = st.selectbox("Sensor", list(sensor_options), key="mission_sensor")
        sensor_id = sensor_options[selected_name]
        sensor = get_sensor(sensor_id)
        st.caption(sensor.description)

        st.markdown("**AOI**")
        latitude = st.number_input(
            "Latitude", -90.0, 90.0, DEFAULT_LATITUDE,
            key="aoi_latitude", format="%.6f",
            help="AOI center latitude.",
        )
        longitude = st.number_input(
            "Longitude", -180.0, 180.0, DEFAULT_LONGITUDE,
            key="aoi_longitude", format="%.6f",
            help="AOI center longitude.",
        )
        area_size = st.slider(
            "Area size (deg)", 0.01, 0.20, 0.05,
            key="aoi_area_size", step=0.01,
            help="Approximate AOI side length in degrees.",
        )
        st.caption("Coordinates can be entered manually or selected on the map.")

        st.markdown("**Temporal**")
        start_date = st.date_input(
            "Start", value=date(2026, 1, 1), key="mission_start_date"
        )
        end_date = st.date_input(
            "End", value=date.today(), key="mission_end_date"
        )

        st.markdown("**Scene Filter**")
        max_cloud_cover = st.slider(
            "Max cloud cover", 0, 100, 20, 1,
            format="%d%%", key="mission_cloud",
        )
        search_clicked = st.button(
            "Search Scenes", type="primary",
            use_container_width=True, key="mission_search",
        )

    return {
        "sensor_id": sensor_id,
        "sensor": sensor,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "area_size": float(area_size),
        "start_date": start_date,
        "end_date": end_date,
        "max_cloud_cover": int(max_cloud_cover),
        "search_clicked": search_clicked,
    }


def render_summary(config: dict, scene_count: int) -> None:
    metrics = [
        ("AOI", f'{config["latitude"]:.4f}, {config["longitude"]:.4f}'),
        ("Area", f'{config["area_size"]:.2f}° × {config["area_size"]:.2f}°'),
        ("Time Window", f'{config["start_date"]:%Y-%m-%d} → {config["end_date"]:%Y-%m-%d}'),
        ("Cloud", f'{config["max_cloud_cover"]}% max'),
        ("Scenes", str(scene_count)),
        ("Status", status_badge("Active", "ready")),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="mission-metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value-small">{value}</div></div>',
                unsafe_allow_html=True,
            )
