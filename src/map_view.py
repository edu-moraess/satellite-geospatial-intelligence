from __future__ import annotations

from typing import Any, Iterable

import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium

from src.aoi import apply_map_click_to_center


# ============================================================
# BASEMAPS
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
            "https://{s}.tile.opentopomap.org/"
            "{z}/{x}/{y}.png"
        ),
        "attr": "OpenTopoMap",
    },
}


# ============================================================
# HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(
    bbox: Iterable[float] | None,
) -> tuple[float, float, float, float] | None:

    if bbox is None:
        return None

    try:
        values = list(bbox)
    except TypeError:
        return None

    if len(values) != 4:
        return None

    try:
        min_lon = float(values[0])
        min_lat = float(values[1])
        max_lon = float(values[2])
        max_lat = float(values[3])
    except (TypeError, ValueError):
        return None

    return (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )


def resolve_slider_default(
    options: list,
    stored_value: Any = None,
    preferred_value: Any = None,
):
    """
    Return a value that is guaranteed to exist in options.

    Prevents stale Streamlit session state from causing:
        ValueError: X is not in iterable
    """

    if not options:
        raise ValueError(
            "resolve_slider_default: options must not be empty."
        )

    if stored_value in options:
        return stored_value

    if preferred_value in options:
        return preferred_value

    return options[len(options) // 2]


# ============================================================
# AOI
# ============================================================

def add_aoi(
    fmap: folium.Map,
    latitude: float,
    longitude: float,
    area_size: float,
) -> None:

    lat = _safe_float(latitude)
    lon = _safe_float(longitude)

    size = max(
        _safe_float(area_size),
        0.001,
    )

    half = size / 2

    bounds = [
        [lat - half, lon - half],
        [lat + half, lon + half],
    ]

    aoi_group = folium.FeatureGroup(
        name="Analysis Area",
        show=True,
    )

    folium.Rectangle(
        bounds=bounds,
        color="#64748B",
        weight=2,
        fill=True,
        fill_color="#64748B",
        fill_opacity=0.08,
        tooltip="Analysis Area",
        popup=(
            "<b>Analysis Area</b><br>"
            f"Latitude: {lat:.6f}<br>"
            f"Longitude: {lon:.6f}<br>"
            f"Area size: {size:.3f}°"
        ),
    ).add_to(aoi_group)

    folium.CircleMarker(
        location=[
            lat,
            lon,
        ],
        radius=5,
        color="#FFFFFF",
        weight=2,
        fill=True,
        fill_color="#475569",
        fill_opacity=1,
        tooltip="Selected location",
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

    normalized = _normalize_bbox(bbox)

    if normalized is None:
        return

    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = normalized

    footprint = folium.FeatureGroup(
        name="Scene Footprint",
        show=True,
    )

    folium.Rectangle(
        bounds=[
            [min_lat, min_lon],
            [max_lat, max_lon],
        ],
        color="#94A3B8",
        weight=2,
        fill=True,
        fill_color="#94A3B8",
        fill_opacity=0.03,
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
        name="Selected Scene",
        show=True,
    )

    folium.Marker(
        location=[
            _safe_float(latitude),
            _safe_float(longitude),
        ],
        tooltip="Selected Sentinel-2 scene",
        popup=folium.Popup(
            popup_html,
            max_width=400,
        ),
        icon=folium.Icon(
            color="gray",
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

    lat = _safe_float(latitude)
    lon = _safe_float(longitude)

    fmap = folium.Map(
        location=[
            lat,
            lon,
        ],
        zoom_start=int(zoom_start),
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # ========================================================
    # BASEMAPS
    # ========================================================

    for name, config in DEFAULT_TILES.items():
        folium.TileLayer(
            tiles=config["tiles"],
            attr=config["attr"],
            name=name,
            overlay=False,
            control=True,
            show=name == map_style,
        ).add_to(fmap)

    # ========================================================
    # AOI
    # ========================================================

    add_aoi(
        fmap=fmap,
        latitude=lat,
        longitude=lon,
        area_size=area_size,
    )

    # ========================================================
    # SCENE FOOTPRINT
    # ========================================================

    if bbox is not None:
        add_scene_footprint(
            fmap=fmap,
            bbox=bbox,
            label=(
                "Sentinel-2 · "
                f"{scene_id or 'Scene'}"
            ),
        )

    # ========================================================
    # SCENE MARKER
    # ========================================================

    if scene_id:
        add_scene_marker(
            fmap=fmap,
            latitude=lat,
            longitude=lon,
            scene_id=scene_id,
            acquisition_date=acquisition_date,
            cloud_cover=cloud_cover,
        )

    # ========================================================
    # FULLSCREEN
    # ========================================================

    plugins.Fullscreen(
        position="topleft",
        title="Expand map",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(fmap)

    # ========================================================
    # MOUSE POSITION
    # ========================================================

    plugins.MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordinates:",
        lat_formatter=(
            "function(num) { "
            "return L.Util.formatNum(num, 6); "
            "}"
        ),
        lng_formatter=(
            "function(num) { "
            "return L.Util.formatNum(num, 6); "
            "}"
        ),
    ).add_to(fmap)

    # ========================================================
    # MEASUREMENT
    # ========================================================

    plugins.MeasureControl(
        position="topleft",
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
        primary_area_unit="sqkilometers",
        secondary_area_unit="sqm",
    ).add_to(fmap)

    # ========================================================
    # DRAWING / AOI SELECTION
    # ========================================================

    plugins.Draw(
        export=True,
        position="topleft",
        draw_options={
            "polyline": False,
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(fmap)

    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(fmap)

    return fmap


# ============================================================
# STREAMLIT MAP RENDERER
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
        height=int(height),
        returned_objects=[
            "last_clicked",
            "bounds",
            "zoom",
            "center",
            "last_object_clicked",
            "all_drawings",
        ],
        key=key,
    )

    if map_state is None:
        return {}

    return map_state


# ============================================================
# MAP CLICK HANDLER
# ============================================================

def _handle_map_click(
    map_state: dict[str, Any],
    current_latitude: float,
    current_longitude: float,
) -> bool:
    """
    Apply a new map click to the AOI session state.

    Returns True only when the AOI center changed.

    A fingerprint is stored to prevent the same
    streamlit-folium click event from causing a rerun loop.
    """

    last_clicked = map_state.get("last_clicked")

    if not isinstance(last_clicked, dict):
        return False

    selected_lat = last_clicked.get("lat")
    selected_lon = last_clicked.get("lng")

    if selected_lat is None or selected_lon is None:
        return False

    clicked_lat = _safe_float(selected_lat, default=float("nan"))
    clicked_lon = _safe_float(selected_lon, default=float("nan"))

    if not (
        clicked_lat == clicked_lat
        and clicked_lon == clicked_lon
    ):
        return False

    click_fingerprint = (
        round(clicked_lat, 7),
        round(clicked_lon, 7),
    )

    previous_click = st.session_state.get(
        "_last_processed_map_click"
    )

    if previous_click == click_fingerprint:
        return False

    result = apply_map_click_to_center(
        latitude=clicked_lat,
        longitude=clicked_lon,
        current_latitude=current_latitude,
        current_longitude=current_longitude,
    )

    # Always record the click fingerprint, even if the
    # coordinate was effectively unchanged.
    st.session_state["_last_processed_map_click"] = (
        click_fingerprint
    )

    if result is None:
        return False

    new_latitude, new_longitude = result

    st.session_state["aoi_latitude"] = new_latitude
    st.session_state["aoi_longitude"] = new_longitude
    st.session_state["aoi_source"] = "map"

    return True


# ============================================================
# PROFESSIONAL MAP PANEL
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

    st.subheader("Geospatial Operations Center")

    st.caption(
        "Select a location directly on the map or use the "
        "AOI controls in the sidebar."
    )

    # ========================================================
    # MAP CONTROLS
    # ========================================================

    control_col1, control_col2, control_col3 = st.columns(3)

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
        height_options = [
            500,
            600,
            650,
            700,
            800,
        ]

        height_state_key = f"{key}_height"

        default_height = resolve_slider_default(
            height_options,
            stored_value=st.session_state.get(
                height_state_key
            ),
            preferred_value=650,
        )

        map_height = st.select_slider(
            "Map height",
            options=height_options,
            value=default_height,
            key=height_state_key,
        )

    st.caption(
        "Click the map to move the analysis area. "
        "Use the drawing tools when you need a custom polygon "
        "or rectangle."
    )

    # ========================================================
    # MAP
    # ========================================================

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

    # ========================================================
    # MAP CLICK
    # ========================================================

    if map_state:
        changed = _handle_map_click(
            map_state=map_state,
            current_latitude=latitude,
            current_longitude=longitude,
        )

        if changed:
            st.rerun()

        # ====================================================
        # SELECTED COORDINATE
        # ====================================================

        last_clicked = map_state.get(
            "last_clicked"
        )

        if isinstance(last_clicked, dict):
            selected_lat = last_clicked.get("lat")
            selected_lon = last_clicked.get("lng")

            if (
                selected_lat is not None
                and selected_lon is not None
            ):
                st.caption(
                    "Selected location · "
                    f"{float(selected_lat):.6f}, "
                    f"{float(selected_lon):.6f}"
                )

        # ====================================================
        # CURRENT MAP CENTER
        # ====================================================

        center = map_state.get("center")
        zoom = map_state.get("zoom")

        if isinstance(center, dict):
            center_lat = center.get("lat")
            center_lon = center.get("lng")

            if (
                center_lat is not None
                and center_lon is not None
            ):
                center_text = (
                    f"Map center · "
                    f"{float(center_lat):.6f}, "
                    f"{float(center_lon):.6f}"
                )

                if zoom is not None:
                    center_text += (
                        f" · Zoom {int(zoom)}"
                    )

                st.caption(center_text)

        # ====================================================
        # DRAWN AOI
        # ====================================================

        drawings = map_state.get(
            "all_drawings"
        )

        if drawings:
            if isinstance(drawings, dict):
                features = drawings.get(
                    "features",
                    [],
                )

                count = (
                    len(features)
                    if isinstance(features, list)
                    else 0
                )

            elif isinstance(drawings, list):
                count = len(drawings)

            else:
                count = 0

            if count:
                st.caption(
                    f"Custom AOI · {count} geometry"
                    + ("ies" if count != 1 else "")
                )

    return map_state