"""
Satellite image visualization utilities.

Creates:

- Natural RGB
- False Color

Sentinel-2 mapping:

RGB:
    B04 = Red
    B03 = Green
    B02 = Blue

False Color:
    B08 = NIR
    B04 = Red
    B03 = Green
"""

from __future__ import annotations

import numpy as np

from .geospatial import (
    normalize_image,
    stack_bands,
)


# ============================================================
# PREPARE BAND
# ============================================================

def prepare_band(
    band,
):
    """
    Normalize one band for visualization.
    """

    return normalize_image(
        band
    )


# ============================================================
# CREATE RGB
# ============================================================

def create_rgb(
    blue,
    green,
    red,
):
    """
    Create a Natural Color RGB image.

    Input:
        blue  -> B02
        green -> B03
        red   -> B04

    Output:
        uint8 RGB image
    """

    blue = prepare_band(
        blue
    )

    green = prepare_band(
        green
    )

    red = prepare_band(
        red
    )

    rgb = stack_bands(
        [
            red,
            green,
            blue,
        ]
    )

    # --------------------------------------------------------
    # GAMMA CORRECTION
    # --------------------------------------------------------

    gamma = 1.0 / 2.2

    rgb = np.power(
        np.clip(
            rgb,
            0.0,
            1.0,
        ),
        gamma,
    )

    return (
        rgb * 255
    ).astype(
        np.uint8
    )


# ============================================================
# CREATE FALSE COLOR
# ============================================================

def create_false_color(
    green,
    red,
    nir,
):
    """
    Create a False Color composite.

    Sentinel-2:

        Red channel   -> NIR (B08)
        Green channel -> Red (B04)
        Blue channel  -> Green (B03)
    """

    green = prepare_band(
        green
    )

    red = prepare_band(
        red
    )

    nir = prepare_band(
        nir
    )

    false_color = stack_bands(
        [
            nir,
            red,
            green,
        ]
    )

    gamma = 1.0 / 2.2

    false_color = np.power(
        np.clip(
            false_color,
            0.0,
            1.0,
        ),
        gamma,
    )

    return (
        false_color * 255
    ).astype(
        np.uint8
    )


# ============================================================
# CREATE GRAYSCALE
# ============================================================

def create_grayscale(
    band,
):
    """
    Create a grayscale visualization.
    """

    normalized = prepare_band(
        band
    )

    return (
        normalized * 255
    ).astype(
        np.uint8
    )


# ============================================================
# IMAGE STATISTICS
# ============================================================

def image_statistics(
    image,
):
    """
    Calculate basic statistics for an RGB image.
    """

    image = np.asarray(
        image
    )

    return {
        "minimum": int(
            image.min()
        ),
        "maximum": int(
            image.max()
        ),
        "mean": float(
            image.mean()
        ),
        "height": int(
            image.shape[0]
        ),
        "width": int(
            image.shape[1]
        ),
    }