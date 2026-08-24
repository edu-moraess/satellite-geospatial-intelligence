from __future__ import annotations
from typing import Any, Iterable, Tuple

import streamlit as st
import pydeck as pdk

def render_map_panel(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
    classification=None,
    ndvi=None,
    detections=None,
    key: str = "main_geospatial_map",
) -> dict[str, Any]:

    st.subheader("🗺️ Geospatial Operations Center")
    st.caption("Interactive Earth observation map • Sentinel-2")

    lat, lon = float(latitude), float(longitude)
    
    # Camada base: polígono simples da AOI
    half = max(float(area_size), 0.001) / 2
    aoi_polygon = [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
    ]

    layers = [
        pdk.Layer(
            "TileLayer",
            data=None,
            get_tile_url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            get_tile_size=256,
            picking=False,
            opacity=1.0,
        ),
        pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": aoi_polygon, "name": "AOI"}],
            get_polygon="polygon",
            get_fill_color=[0, 212, 255, 50],
            get_line_color=[0, 212, 255, 200],
            line_width_min_pixels=2,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": [lon, lat], "name": "Center"}],
            get_position="position",
            get_radius=100,
            radius_min_pixels=5,
            get_fill_color=[255, 255, 255, 255],
            pickable=True,
        ),
    ]

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=40, bearing=0)
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
    )

    st.pydeck_chart(deck, width='stretch', key=key)
    return {}