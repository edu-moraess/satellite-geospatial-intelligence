from __future__ import annotations
from typing import Any, Iterable, Tuple
import streamlit as st
import pydeck as pdk

def render_map_panel(
    latitude: float, longitude: float, area_size: float = 0.05,
    bbox: Iterable[float] | None = None, scene_id: str | None = None,
    acquisition_date: str | None = None, cloud_cover: float | None = None,
    classification=None, ndvi=None, detections=None,
    key: str = "main_geospatial_map",
) -> dict[str, Any]:

    st.subheader("🗺️ Geospatial Operations Center")
    st.caption("Interactive Earth observation map • Sentinel-2")
    lat, lon = float(latitude), float(longitude)
    half = max(float(area_size), 0.001) / 2

    # Dados dos Polígonos
    aoi_data = [{"polygon": [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half]], "name": "Default AOI"}]
    point_data = [{"position": [lon, lat], "name": "AOI Center"}]
    footprint_data = []
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        footprint_data = [{"polygon": [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat]], "name": "Scene Footprint"}]

    # Camadas: Satélite por baixo + Polígonos Neon por cima
    layers = [
        # Imagem de Satélite
        pdk.Layer(
            "TileLayer",
            data=None,
            get_tile_url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            get_tile_size=256,
            picking=False,
            opacity=1.0,
        ),
        # Polígono da Cena (Amarelo)
        pdk.Layer(
            "PolygonLayer",
            data=footprint_data,
            get_polygon="polygon",
            get_fill_color=[255, 209, 102, 20],
            get_line_color=[255, 209, 102, 255],
            line_width_min_pixels=2,
            pickable=True,
        ),
        # Polígono do AOI (Ciano Neon)
        pdk.Layer(
            "PolygonLayer",
            data=aoi_data,
            get_polygon="polygon",
            get_fill_color=[0, 212, 255, 60],
            get_line_color=[0, 212, 255, 255],
            line_width_min_pixels=3,
            pickable=True,
        ),
        # Ponto Central (Branco)
        pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="position",
            get_radius=100,
            radius_min_pixels=6,
            get_fill_color=[255, 255, 255, 255],
            pickable=True,
        ),
    ]

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=40, bearing=0)
    
    # tooltip
    tooltip = {"html": "<b>{name}</b>", "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white", "borderRadius": "6px"}}

    # map_style=None é crucial para deixar a imagem de satélite aparecer!
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,
        tooltip=tooltip
    )
    st.pydeck_chart(deck, width='stretch', key=key)
    return {}