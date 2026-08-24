"""
Land Cover Analysis
===================

Utilities for summarizing and visualizing
the classified satellite scene using Plotly.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px

# ============================================================
# CLASS COLORS
# ============================================================

CLASS_COLORS = [
    "#BDBDBD",  # Other (Cinza)
    "#2E7D32",  # Vegetation (Verde)
    "#1976D2",  # Water (Azul)
    "#D84315",  # Built-up (Vermelho/Laranja)
    "#C49A6C",  # Bare soil (Marrom)
]

# ============================================================
# CREATE LAND COVER MAP
# ============================================================

def create_land_cover_figure(classification):
    """
    Create a categorical land-cover map with Plotly.
    """
    classification = np.asarray(classification)

    discrete_colorscale = [
        [0.0, CLASS_COLORS[0]],
        [0.25, CLASS_COLORS[1]],
        [0.50, CLASS_COLORS[2]],
        [0.75, CLASS_COLORS[3]],
        [1.00, CLASS_COLORS[4]],
    ]

    fig = px.imshow(
        classification,
        color_continuous_scale=discrete_colorscale,
        zmin=0,
        zmax=4,
        labels=dict(x="Pixel X", y="Pixel Y", color="Class"),
        aspect="auto"
    )

    fig.update_layout(
        title="Land Cover Classification",
        title_font=dict(color="white", size=18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        coloraxis_colorbar=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Other", "Vegetation", "Water", "Built-up", "Bare Soil"],
            title="Classes",
            title_font_color="white",
            tickfont_color="white",
            outlinewidth=1,
            outlinecolor="white"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

# ============================================================
# AREA ESTIMATE
# ============================================================

def calculate_area_km2(classification, pixel_size_meters=10.0):
    """
    Estimate area per class.

    Default:
        10 m Sentinel-2 grid.
    """

    classification = np.asarray(classification)

    pixel_area_m2 = (
        pixel_size_meters
        * pixel_size_meters
    )

    pixel_area_km2 = (
        pixel_area_m2
        / 1_000_000.0
    )

    classes = {
        0: "Other",
        1: "Vegetation",
        2: "Water",
        3: "Built-up",
        4: "Bare Soil",
    }

    result = {}

    for class_id, name in (
        classes.items()
    ):

        pixels = np.sum(
            classification
            == class_id
        )

        result[name] = (
            float(pixels)
            * pixel_area_km2
        )

    return result