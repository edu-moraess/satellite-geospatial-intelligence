"""
Land Cover Analysis
===================

Utilities for summarizing and visualizing
the classified satellite scene.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# CLASS COLORS
# ============================================================

CLASS_COLORS = [
    "#BDBDBD",  # Other
    "#2E7D32",  # Vegetation
    "#1976D2",  # Water
    "#D84315",  # Built-up
    "#C49A6C",  # Bare soil
]


# ============================================================
# CREATE LAND COVER MAP
# ============================================================

def create_land_cover_figure(
    classification,
):
    """
    Create a categorical land-cover map.
    """

    classification = np.asarray(
        classification
    )

    cmap = ListedColormap(
        CLASS_COLORS
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        classification,
        cmap=cmap,
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )

    axis.set_title(
        "Land Cover Classification",
        fontsize=16,
    )

    axis.set_xlabel(
        "Pixel X"
    )

    axis.set_ylabel(
        "Pixel Y"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    colorbar.ax.set_yticklabels(
        [
            "Other",
            "Vegetation",
            "Water",
            "Built-up",
            "Bare Soil",
        ]
    )

    figure.tight_layout()

    return figure


# ============================================================
# AREA ESTIMATE
# ============================================================

def calculate_area_km2(
    classification,
    pixel_size_meters=10.0,
):
    """
    Estimate area per class.

    Default:
        10 m Sentinel-2 grid.

    Note:
        This is an approximation for the selected
        local AOI. A geodesic/CRS-aware area
        calculation can be added later.
    """

    classification = np.asarray(
        classification
    )

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