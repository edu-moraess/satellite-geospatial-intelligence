from __future__ import annotations
from typing import Any, Iterable, Tuple, List, Optional

import streamlit as st
import pydeck as pdk
import numpy as np


def _normalize_bbox(bbox) -> Tuple[float, float, float, float] | None:
    if bbox is None: return None
    try:
        values = list(bbox)
        if len(values) != 4: return None
        return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    except: return None

def _pixel_to_latlon(bbox, width, height, x, y):
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (x / width) * (max_lon - min_lon)
    lat = max_lat - (y / height) * (max_lat - min_lat)
    return lon, lat

def create_geospatial_map(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
    classification: np.ndarray | None = None,
    ndvi: np.ndarray | None = None,
    detections: list | None = None,
) -> pdk.Deck:
    
    lat, lon = float(latitude), float(longitude)
    half = max(float(area_size), 0.001) / 2

    # Camada base: Retângulo da AOI
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
        )
    ]

    # -----------------------------
    # CAMADA 1: HEATMAP DE NDVI (Verde)
    # -----------------------------
    if ndvi is not None and bbox:
        min_lon, min_lat, max_lon, max_lat = _normalize_bbox(bbox)
        height, width = ndvi.shape
        heatmap_data = []
        
        # Amostra para não sobrecarregar o mapa
        step_x = max(1, width // 50)
        step_y = max(1, height // 50)
        
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                # 1 = Vegetação, -1 = Água/Solo
                value = float(ndvi[y, x])
                if np.isfinite(value):
                    plon, plat = _pixel_to_latlon((min_lon, min_lat, max_lon, max_lat), width, height, x, y)
                    heatmap_data.append({
                        "position": [plon, plat],
                        "weight": value
                    })
        
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=heatmap_data,
            get_position="position",
            get_weight="weight",
            radius_pixels=20,
            intensity=1.5,
            threshold=0.5,
            color_range=[[0, 200, 0], [0, 255, 0]]  # Cores neon esverdeadas
        ))

    # -----------------------------
    # CAMADA 2: CLASSIFICAÇÃO (Polígonos)
    # -----------------------------
    if classification is not None and bbox:
        # Simplificação: mostra "Built-up" (Vermelho) e "Vegetation" (Verde)
        min_lon, min_lat, max_lon, max_lat = _normalize_bbox(bbox)
        height, width = classification.shape
        step_x = max(1, width // 40)
        step_y = max(1, height // 40)
        
        class_data = []
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                val = int(classification[y, x])
                color = [0,0,0]
                if val == 1: color = [46, 125, 50, 200] # Vegetation
                elif val == 2: color = [25, 118, 210, 200] # Water
                elif val == 3: color = [216, 67, 21, 200] # Built-up
                elif val == 4: color = [196, 154, 108, 200] # Bare soil
                
                if color != [0,0,0]:
                    plon, plat = _pixel_to_latlon((min_lon, min_lat, max_lon, max_lat), width, height, x, y)
                    class_data.append({
                        "position": [plon, plat],
                        "size": (area_size / 40) * 50000, # Tamanho proporcional
                        "color": color
                    })

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=class_data,
            get_position="position",
            get_radius="size",
            get_fill_color="color",
            stroked=False,
            pickable=True
        ))

    # -----------------------------
    # CAMADA 3: DETECÇÕES DE IA (Bounding Boxes)
    # -----------------------------
    if detections and bbox:
        min_lon, min_lat, max_lon, max_lat = _normalize_bbox(bbox)
        height, width = ndvi.shape if ndvi is not None else 512, 512
        
        detection_data = []
        for det in detections:
            x_min, y_min, w, h = det["bbox"]
            # Converte a caixa em 4 pontos para um polígono
            p1 = _pixel_to_latlon((min_lon, min_lat, max_lon, max_lat), width, height, x_min, y_min)
            p2 = _pixel_to_latlon((min_lon, min_lat, max_lon, max_lat), width, height, x_min + w, y_min)
            p3 = _pixel_to_latlon((min_lon, min_lat, max_lon, max_lat), width, height, x_min + w, y_min + h)
            p4 = _pixel_to_latlon((min_lon, min_lat, max_lon, max_lat), width, height, x_min, y_min + h)
            
            detection_data.append({
                "polygon": [p1, p2, p3, p4],
                "name": f"Det: {det['class']} ({det['confidence']:.2f})"
            })

        layers.append(pdk.Layer(
            "PolygonLayer",
            data=detection_data,
            get_polygon="polygon",
            get_fill_color=[255, 0, 128, 50],
            get_line_color=[255, 0, 128, 255],
            line_width_min_pixels=2,
            pickable=True
        ))

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=40, bearing=0)
    tooltip = {
        "html": "<b>{name}</b>",
        "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white", "borderRadius": "6px", "fontFamily": "Arial, sans-serif", "padding": "8px"}
    }

    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style="dark", tooltip=tooltip)


def render_map_panel(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
    classification: np.ndarray | None = None,
    ndvi: np.ndarray | None = None,
    detections: list | None = None,
    key: str = "main_geospatial_map",
) -> dict[str, Any]:

    st.subheader("🗺️ Geospatial Operations Center")
    st.caption("Interactive Earth observation map • Sentinel-2 • AOI")

    # Controles do Mapa
    c1, c2 = st.columns(2)
    with c1: zoom_start = st.slider("Zoom", 3, 18, 12, key=f"{key}_zoom")
    with c2: pitch = st.slider("3D Tilt", 0, 60, 40, key=f"{key}_pitch")

    # Cria o mapa com as camadas de dados
    deck = create_geospatial_map(
        latitude=latitude, longitude=longitude, area_size=area_size,
        bbox=bbox, scene_id=scene_id, acquisition_date=acquisition_date, cloud_cover=cloud_cover,
        classification=classification, ndvi=ndvi, detections=detections
    )

    deck.initial_view_state.zoom = zoom_start
    deck.initial_view_state.pitch = pitch

    # Renderiza o mapa
    st.pydeck_chart(deck, use_container_width=True, key=key)

    return {}