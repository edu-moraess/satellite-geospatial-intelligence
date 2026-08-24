"""
Satellite visualization.

Phase 1:
- Natural RGB
- False Color
"""

import numpy as np

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
    Sentinel-2 natural RGB.

    R = B04
    G = B03
    B = B02
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

    return np.dstack(
        [
            red,
            green,
            blue,
        ]
    )


# ============================================================
# FALSE COLOR
# ============================================================

def create_false_color(
    green_path,
    red_path,
    nir_path,
):
    """
    False color composite.

    R = NIR
    G = RED
    B = GREEN
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

    return np.dstack(
        [
            nir,
            red,
            green,
        ]
    )