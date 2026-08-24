from __future__ import annotations
from typing import Any, Iterable, Tuple
import streamlit as st
import pydeck as pdk
import numpy as np

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

    # Base Layers (Poligonos)
    aoi_data = [{"polygon": [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half]], "name": "Default AOI"}]
    point_data = [{"position": [lon, lat], "name": "AOI Center"}]
    footprint_data = []
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        footprint_data = [{"polygon": [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat]], "name": "Scene Footprint"}]

    layers = [
        pdk.Layer("TileLayer", data=None, get_tile_url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", get_tile_size=256, picking=False, opacity=1.0),
        pdk.Layer("PolygonLayer", data=footprint_data, get_polygon="polygon", get_fill_color=[255, 209, 102, 20], get_line_color=[255, 209, 102, 255], line_width_min_pixels=2, pickable=True),
        pdk.Layer("PolygonLayer", data=aoi_data, get_polygon="polygon", get_fill_color=[0, 212, 255, 60], get_line_color=[0, 212, 255, 255], line_width_min_pixels=3, pickable=True),
        pdk.Layer("ScatterplotLayer", data=point_data, get_position="position", get_radius=100, radius_min_pixels=6, get_fill_color=[255, 255, 255, 255], pickable=True),
    ]

    # CAMADA 1: HEATMAP DE NDVI (Verde)
    if ndvi is not None and bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        h, w = ndvi.shape
        step_x = max(1, w // 50)
        step_y = max(1, h // 50)
        heatmap_data = []
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                val = float(ndvi[y, x])
                if np.isfinite(val):
                    plon = min_lon + (x / w) * (max_lon - min_lon)
                    plat = max_lat - (y / h) * (max_lat - min_lat)
                    heatmap_data.append({"position": [plon, plat], "weight": val})
        if heatmap_data:
            layers.append(pdk.Layer("HeatmapLayer", data=heatmap_data, get_position="position", get_weight="weight", radius_pixels=20, intensity=1.5, threshold=0.5, color_range=[[0, 150, 0], [0, 255, 0]]))

    # CAMADA 2: CLASSIFICAÇÃO (Pontos coloridos)
    if classification is not None and bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        h, w = classification.shape
        step_x = max(1, w // 40)
        step_y = max(1, h // 40)
        class_data = []
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                val = int(classification[y, x])
                color = [0,0,0]
                if val == 1: color = [46, 125, 50, 200]
                elif val == 2: color = [25, 118, 210, 200]
                elif val == 3: color = [216, 67, 21, 200]
                elif val == 4: color = [196, 154, 108, 200]
                if color != [0,0,0]:
                    plon = min_lon + (x / w) * (max_lon - min_lon)
                    plat = max_lat - (y / h) * (max_lat - min_lat)
                    class_data.append({"position": [plon, plat], "size": (area_size / 40) * 50000, "color": color})
        if class_data:
            layers.append(pdk.Layer("ScatterplotLayer", data=class_data, get_position="position", get_radius="size", get_fill_color="color", stroked=False, pickable=True))

    # CAMADA 3: DETECÇÕES IA (Bounding Boxes)
    if detections and bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        h, w = ndvi.shape if ndvi is not None else 512, 512
        det_data = []
        for det in detections:
            x_min, y_min, bw, bh = det["bbox"]
            p1 = [min_lon + (x_min/w)*(max_lon-min_lon), max_lat - (y_min/h)*(max_lat-min_lat)]
            p2 = [min_lon + ((x_min+bw)/w)*(max_lon-min_lon), max_lat - (y_min/h)*(max_lat-min_lat)]
            p3 = [min_lon + ((x_min+bw)/w)*(max_lon-min_lon), max_lat - ((y_min+bh)/h)*(max_lat-min_lat)]
            p4 = [min_lon + (x_min/w)*(max_lon-min_lon), max_lat - ((y_min+bh)/h)*(max_lat-min_lat)]
            det_data.append({"polygon": [p1, p2, p3, p4], "name": f"Det: {det['class']}"})
        if det_data:
            layers.append(pdk.Layer("PolygonLayer", data=det_data, get_polygon="polygon", get_fill_color=[255, 0, 128, 50], get_line_color=[255, 0, 128, 255], line_width_min_pixels=2, pickable=True))

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=40, bearing=0)
    tooltip = {"html": "<b>{name}</b>", "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white", "borderRadius": "6px"}}
    
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip)
    st.pydeck_chart(deck, width='stretch', key=key)
    return {}