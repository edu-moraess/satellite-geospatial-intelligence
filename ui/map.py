from __future__ import annotations

from typing import Any

import folium
import streamlit as st

from streamlit_folium import st_folium


def create_gis_map(
    latitude: float,
    longitude: float,
    area_size: float,
    title: str = "Geospatial Map",
    show_aoi: bool = True,
    show_controls: bool = True,
) -> Any:
    """
    Create an interactive GIS map.

    Features:
        - Pan
        - Zoom
        - AOI visualization
        - Layer control
        - Scale
        - Coordinate interaction
    """

    fmap = folium.Map(
        location=[
            latitude,
            longitude,
        ],
        zoom_start=12,
        control_scale=True,
        tiles=None,
    )

    # ---------------------------------------------------------
    # Base layers
    # ---------------------------------------------------------

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(fmap)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/"
            "tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        control=True,
    ).add_to(fmap)

    folium.TileLayer(
        tiles=(
            "https://{s}.tile.opentopomap.org/"
            "{z}/{x}/{y}.png"
        ),
        attr="OpenTopoMap",
        name="Terrain",
        control=True,
    ).add_to(fmap)

    # ---------------------------------------------------------
    # AOI
    # ---------------------------------------------------------

    if show_aoi:

        half = area_size / 2

        south = latitude - half
        north = latitude + half

        west = longitude - half
        east = longitude + half

        bounds = [
            [south, west],
            [north, east],
        ]

        aoi_group = folium.FeatureGroup(
            name="Area of Interest",
            show=True,
        )

        folium.Rectangle(
            bounds=bounds,
            popup=(
                f"AOI<br>"
                f"Latitude: {latitude:.6f}<br>"
                f"Longitude: {longitude:.6f}"
            ),
            tooltip="Area of Interest",
            fill=True,
            fill_opacity=0.08,
            weight=2,
        ).add_to(aoi_group)

        folium.Marker(
            location=[
                latitude,
                longitude,
            ],
            tooltip="AOI Center",
            popup=(
                f"Latitude: {latitude:.6f}<br>"
                f"Longitude: {longitude:.6f}"
            ),
        ).add_to(aoi_group)

        aoi_group.add_to(fmap)

    # ---------------------------------------------------------
    # Map tools
    # ---------------------------------------------------------

    if show_controls:

        folium.LayerControl(
            collapsed=False,
        ).add_to(fmap)

        folium.LatLngPopup().add_to(fmap)

        folium.plugins.Fullscreen(
            position="topleft",
        ).add_to(fmap)

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            margin-bottom:0.55rem;
            color:#71838C;
            font-size:0.72rem;
            text-transform:uppercase;
            letter-spacing:0.08em;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st_folium(
        fmap,
        width=None,
        height=560,
        returned_objects=[
            "last_clicked",
            "bounds",
            "zoom",
        ],
    )