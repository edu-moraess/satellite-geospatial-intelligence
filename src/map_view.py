from __future__ import annotations
from typing import Any, Iterable, Tuple

import streamlit as st
import pydeck as pdk


def _normalize_bbox(bbox) -> Tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        values = list(bbox)
        if len(values) != 4:
            return None
        return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    except:
        return None


def create_geospatial_map(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
) -> pdk.Deck:
    
    lat, lon = float(latitude), float(longitude)
    half = max(float(area_size), 0.001) / 2

    # Dados das camadas
    aoi_data = [{
        "polygon": [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half]],
        "name": "Default AOI"
    }]

    point_data = [{"position": [lon, lat], "name": "AOI Center"}]

    footprint_data = []
    if bbox:
        min_lon, min_lat, max_lon, max_lat = _normalize_bbox(bbox)
        footprint_data = [{
            "polygon": [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat]],
            "name": "Scene Footprint"
        }]

    # Camadas do Deck.gl
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
            data=aoi_data,
            get_polygon="polygon",
            get_fill_color=[0, 212, 255, 50],
            get_line_color=[0, 212, 255, 200],
            line_width_min_pixels=2,
            pickable=True,
        ),
        pdk.Layer(
            "PolygonLayer",
            data=footprint_data,
            get_polygon="polygon",
            get_fill_color=[255, 209, 102, 20],
            get_line_color=[255, 209, 102, 200],
            line_width_min_pixels=2,
            dash_array=[6, 6],
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="position",
            get_fill_color=[255, 255, 255, 255],
            get_radius=100,
            radius_min_pixels=5,
            pickable=True,
        ),
    ]

    # Configuração da câmera 3D
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=12,
        pitch=40,
        bearing=0,
    )

    # Tooltip
    tooltip = {
        "html": "<b>{name}</b>",
        "style": {
            "backgroundColor": "rgba(15, 23, 42, 0.9)",
            "color": "white",
            "borderRadius": "6px",
            "fontFamily": "Arial, sans-serif",
            "padding": "8px",
        }
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        tooltip=tooltip,
    )


def render_map_panel(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
    key: str = "main_geospatial_map",
) -> dict[str, Any]:

    st.subheader("🗺️ Geospatial Operations Center")
    st.caption("Interactive Earth observation map • Sentinel-2 • AOI")

    c1, c2 = st.columns(2)
    with c1:
        zoom_start = st.slider("Zoom", 3, 18, 12, key=f"{key}_zoom")
    with c2:
        pitch = st.slider("3D Tilt", 0, 60, 40, key=f"{key}_pitch")

    deck = create_geospatial_map(
        latitude=latitude, longitude=longitude, area_size=area_size,
        bbox=bbox, scene_id=scene_id, acquisition_date=acquisition_date, cloud_cover=cloud_cover
    )

    deck.initial_view_state.zoom = zoom_start
    deck.initial_view_state.pitch = pitch

    # Renderiza o mapa (o retorno do pydeck não deve ser iterado, por isso removemos a lógica de 'coordinate')
    st.pydeck_chart(deck, use_container_width=True, key=key)

    # Retornamos um dicionário vazio por padrão (sem quebrar a interface)
    return {}