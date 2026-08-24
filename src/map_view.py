from __future__ import annotations

from typing import Any, Iterable

import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# MAP STYLE
# ============================================================

DEFAULT_TILES = {
    "Satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "Esri World Imagery",
    },
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": "OpenStreetMap",
    },
    "Terrain": {
        "tiles": (
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        ),
        "attr": "OpenTopoMap",
    },
}


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(
    bbox: Iterable[float] | None,
) -> tuple[float, float, float, float] | None:
    """
    Normalize bbox into:

        min_lon, min_lat, max_lon, max_lat
    """

    if bbox is None:
        return None

    values = list(bbox)

    if len(values) != 4:
        return None

    min_lon = _safe_float(values[0])
    min_lat = _safe_float(values[1])
    max_lon = _safe_float(values[2])
    max_lat = _safe_float(values[3])

    return (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )


# ============================================================
# AOI
# ============================================================

def add_aoi(
    fmap: folium.Map,
    latitude: float,
    longitude: float,
    area_size: float,
) -> None:
    """
    Add the Area of Interest to the map.
    """

    lat = _safe_float(latitude)
    lon = _safe_float(longitude)
    size = max(_safe_float(area_size), 0.001)

    half = size / 2

    bounds = [
        [lat - half, lon - half],
        [lat + half, lon + half],
    ]

    aoi_group = folium.FeatureGroup(
        name="📍 Area of Interest",
        show=True,
    )

    folium.Rectangle(
        bounds=bounds,
        color="#00D4FF",
        weight=2,
        fill=True,
        fill_color="#00D4FF",
        fill_opacity=0.08,
        tooltip="Area of Interest",
        popup=(
            f"<b>Area of Interest</b><br>"
            f"Latitude: {lat:.6f}<br>"
            f"Longitude: {lon:.6f}<br>"
            f"Area size: {size:.3f}°"
        ),
    ).add_to(aoi_group)

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color="#FFFFFF",
        weight=2,
        fill=True,
        fill_color="#00D4FF",
        fill_opacity=1,
        tooltip="AOI Center",
    ).add_to(aoi_group)

    aoi_group.add_to(fmap)


# ============================================================
# SCENE FOOTPRINT
# ============================================================

def add_scene_footprint(
    fmap: folium.Map,
    bbox: Iterable[float] | None,
    label: str = "Sentinel-2 Scene",
) -> None:
    """
    Draw the selected scene footprint when a bbox is available.
    """

    normalized = _normalize_bbox(bbox)

    if normalized is None:
        return

    min_lon, min_lat, max_lon, max_lat = normalized

    footprint = folium.FeatureGroup(
        name="🛰️ Scene Footprint",
        show=True,
    )

    folium.Rectangle(
        bounds=[
            [min_lat, min_lon],
            [max_lat, max_lon],
        ],
        color="#FFD166",
        weight=2,
        fill=True,
        fill_color="#FFD166",
        fill_opacity=0.04,
        dash_array="6,6",
        tooltip=label,
    ).add_to(footprint)

    footprint.add_to(fmap)


# ============================================================
# SCENE MARKER
# ============================================================

def add_scene_marker(
    fmap: folium.Map,
    latitude: float,
    longitude: float,
    scene_id: str,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
) -> None:
    """
    Add a professional scene marker.
    """

    popup_lines = [
        "<b>Sentinel-2 Scene</b>",
        f"<b>ID:</b> {scene_id}",
    ]

    if acquisition_date:
        popup_lines.append(
            f"<b>Acquisition:</b> {acquisition_date}"
        )

    if cloud_cover is not None:
        popup_lines.append(
            f"<b>Cloud:</b> {float(cloud_cover):.2f}%"
        )

    popup_html = "<br>".join(popup_lines)

    marker_group = folium.FeatureGroup(
        name="🛰️ Selected Scene",
        show=True,
    )

    folium.Marker(
        location=[
            _safe_float(latitude),
            _safe_float(longitude),
        ],
        tooltip="Selected Sentinel-2 Scene",
        popup=folium.Popup(
            popup_html,
            max_width=400,
        ),
        icon=folium.Icon(
            color="blue",
            icon="satellite",
            prefix="fa",
        ),
    ).add_to(marker_group)

    marker_group.add_to(fmap)


# ============================================================
# MAP CREATION
# ============================================================

def create_geospatial_map(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    zoom_start: int = 12,
    map_style: str = "Satellite",
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
) -> folium.Map:
    """
    Create the main interactive geospatial map.

    Features:
        - Satellite basemap
        - AOI
        - Scene footprint
        - Scene marker
        - Layer control
        - Fullscreen
        - Mouse position
        - Measurement tools
        - Drawing tools
        - Scale
    """

    lat = _safe_float(latitude)
    lon = _safe_float(longitude)

    style = DEFAULT_TILES.get(
        map_style,
        DEFAULT_TILES["Satellite"],
    )

    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # --------------------------------------------------------
    # BASEMAPS
    # --------------------------------------------------------

    for name, config in DEFAULT_TILES.items():

        folium.TileLayer(
            tiles=config["tiles"],
            attr=config["attr"],
            name=f"🗺️ {name}",
            overlay=False,
            control=True,
            show=(
                name == map_style
            ),
        ).add_to(fmap)

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    add_aoi(
        fmap=fmap,
        latitude=lat,
        longitude=lon,
        area_size=area_size,
    )

    # --------------------------------------------------------
    # SCENE
    # --------------------------------------------------------

    if bbox is not None:

        add_scene_footprint(
            fmap=fmap,
            bbox=bbox,
            label=(
                f"Sentinel-2 • "
                f"{scene_id or 'Scene'}"
            ),
        )

    if scene_id:

        add_scene_marker(
            fmap=fmap,
            latitude=lat,
            longitude=lon,
            scene_id=scene_id,
            acquisition_date=acquisition_date,
            cloud_cover=cloud_cover,
        )

    # --------------------------------------------------------
    # FULLSCREEN
    # --------------------------------------------------------

    plugins.Fullscreen(
        position="topleft",
        title="Expand map",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(fmap)

    # --------------------------------------------------------
    # MOUSE POSITION
    # --------------------------------------------------------

    plugins.MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordinates:",
        lat_formatter=(
            "function(num) { return L.Util.formatNum(num, 6); }"
        ),
        lng_formatter=(
            "function(num) { return L.Util.formatNum(num, 6); }"
        ),
    ).add_to(fmap)

    # --------------------------------------------------------
    # MEASUREMENT TOOL
    # --------------------------------------------------------

    plugins.MeasureControl(
        position="topleft",
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
        primary_area_unit="sqkilometers",
        secondary_area_unit="sqm",
    ).add_to(fmap)

    # --------------------------------------------------------
    # DRAWING TOOL
    # --------------------------------------------------------

    plugins.Draw(
        export=True,
        position="topleft",
        draw_options={
            "polyline": True,
            "polygon": True,
            "rectangle": True,
            "circle": True,
            "marker": True,
            "circlemarker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(fmap)

    # --------------------------------------------------------
    # LAYER CONTROL
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(fmap)

    # --------------------------------------------------------
    # MAP INFO
    # --------------------------------------------------------

    info = folium.Element(
        """
        <div style="
            position: fixed;
            top: 12px;
            left: 52px;
            z-index: 9999;
            background: rgba(15, 23, 42, 0.92);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        ">
            <b>GEO INTELLIGENCE MAP</b>
        </div>
        """
    )

    fmap.get_root().html.add_child(info)

    return fmap


# ============================================================
# STREAMLIT RENDERER
# ============================================================

def render_geospatial_map(
    latitude: float,
    longitude: float,
    area_size: float = 0.05,
    zoom_start: int = 12,
    map_style: str = "Satellite",
    bbox: Iterable[float] | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    cloud_cover: float | None = None,
    height: int = 650,
    key: str = "geospatial_map",
) -> dict[str, Any]:
    """
    Render the interactive geospatial map inside Streamlit.

    Returns the map interaction state.
    """

    fmap = create_geospatial_map(
        latitude=latitude,
        longitude=longitude,
        area_size=area_size,
        zoom_start=zoom_start,
        map_style=map_style,
        bbox=bbox,
        scene_id=scene_id,
        acquisition_date=acquisition_date,
        cloud_cover=cloud_cover,
    )

    map_state = st_folium(
        fmap,
        width=None,
        height=height,
        returned_objects=[
            "last_clicked",
            "bounds",
            "zoom",
            "center",
            "last_object_clicked",
        ],
        key=key,
    )

    return map_state


# ============================================================
# STREAMLIT MAP PANEL
# ============================================================

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
    """
    Professional Streamlit map panel.

    This function creates the UI controls around the map.
    """

    st.subheader(
        "🗺️ Geospatial Operations Center"
    )

    st.caption(
        "Interactive Earth observation map • "
        "Sentinel-2 • AOI • Spatial analysis"
    )

    control_col1, control_col2, control_col3 = (
        st.columns(3)
    )

    with control_col1:

        map_style = st.selectbox(
            "Basemap",
            list(DEFAULT_TILES.keys()),
            index=0,
            key=f"{key}_style",
        )

    with control_col2:

        zoom_start = st.slider(
            "Zoom",
            min_value=3,
            max_value=18,
            value=12,
            step=1,
            key=f"{key}_zoom",
        )

    with control_col3:

        map_height = st.select_slider(
            "Map height",
            options=[
                500,
                600,
                700,
                800,
            ],
            value=650,
            key=f"{key}_height",
        )

    map_state = render_geospatial_map(
        latitude=latitude,
        longitude=longitude,
        area_size=area_size,
        zoom_start=zoom_start,
        map_style=map_style,
        bbox=bbox,
        scene_id=scene_id,
        acquisition_date=acquisition_date,
        cloud_cover=cloud_cover,
        height=map_height,
        key=key,
    )

    # --------------------------------------------------------
    # INTERACTION INFORMATION
    # --------------------------------------------------------

    if map_state:

        last_clicked = map_state.get(
            "last_clicked"
        )

        if last_clicked:

            lat = last_clicked.get(
                "lat"
            )

            lon = last_clicked.get(
                "lng"
            )

            if lat is not None and lon is not None:

                st.caption(
                    f"📍 Selected coordinate: "
                    f"{float(lat):.6f}, "
                    f"{float(lon):.6f}"
                )

    return map_state