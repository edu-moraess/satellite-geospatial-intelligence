"""
Satellite visualization utilities.

Phase 1:
- RGB composite
- False Color composite
"""

import numpy as np
import matplotlib.pyplot as plt

from .geospatial import (
    read_band,
    normalize_image,
)


# ============================================================
# RGB
# ============================================================

def create_rgb(
    blue_path,
    green_path,
    red_path,
):
    """
    Create natural RGB composite.

    Sentinel-2:
    B04 -> Red
    B03 -> Green
    B02 -> Blue
    """

    blue, _ = read_band(
        blue_path
    )

    green, _ = read_band(
        green_path
    )

    red, _ = read_band(
        red_path
    )

    blue = normalize_image(
        blue
    )

    green = normalize_image(
        green
    )

    red = normalize_image(
        red
    )

    rgb = np.dstack(
        [
            red,
            green,
            blue,
        ]
    )

    return rgb


# ============================================================
# FALSE COLOR
# ============================================================

def create_false_color(
    green_path,
    red_path,
    nir_path,
):
    """
    Create false-color composite.

    NIR -> Red
    Red -> Green
    Green -> Blue
    """

    green, _ = read_band(
        green_path
    )

    red, _ = read_band(
        red_path
    )

    nir, _ = read_band(
        nir_path
    )

    green = normalize_image(
        green
    )

    red = normalize_image(
        red
    )

    nir = normalize_image(
        nir
    )

    false_color = np.dstack(
        [
            nir,
            red,
            green,
        ]
    )

    return false_color


# ============================================================
# SAVE FIGURE
# ============================================================

def save_image(
    image,
    output_path,
    title,
):
    """
    Save a visualization.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        image
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()