"""
Spectral index calculations.

Supported indices:

NDVI -> vegetation
NDWI -> water
NDBI -> built-up

This module is responsible only for spectral mathematics.
Raster integrity and pixel-grid compatibility are delegated
to src.raster_validation.
"""

from __future__ import annotations

import numpy as np

from src.raster_validation import (
    validate_raster_pair,
)


# ============================================================
# SAFE FLOAT ARRAY
# ============================================================

def _to_float_array(array):
    """
    Convert an input array to float32.
    """

    return np.asarray(
        array,
        dtype=np.float32,
    )


# ============================================================
# NORMALIZED DIFFERENCE
# ============================================================

def normalized_difference(
    band_a,
    band_b,
):
    """
    Generic normalized difference:

        (A - B) / (A + B)

    Invalid divisions are converted to NaN.

    Before performing the pixel-wise operation, both
    rasters are validated and their spatial dimensions
    are checked.
    """

    band_a = _to_float_array(
        band_a
    )

    band_b = _to_float_array(
        band_b
    )

    validate_raster_pair(
        band_a,
        band_b,
        label_a="band A",
        label_b="band B",
    )

    numerator = (
        band_a - band_b
    )

    denominator = (
        band_a + band_b
    )

    result = np.full(
        denominator.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(band_a)
        & np.isfinite(band_b)
        & np.isfinite(denominator)
        & (
            np.abs(denominator)
            > 1e-10
        )
    )

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    result[
        ~np.isfinite(result)
    ] = np.nan

    return result


# ============================================================
# NDVI
# ============================================================

def calculate_ndvi(
    red,
    nir,
):
    """
    NDVI:

        (NIR - RED) / (NIR + RED)
    """

    return normalized_difference(
        nir,
        red,
    )


# ============================================================
# NDWI
# ============================================================

def calculate_ndwi(
    green,
    nir,
):
    """
    NDWI:

        (GREEN - NIR) / (GREEN + NIR)
    """

    return normalized_difference(
        green,
        nir,
    )


# ============================================================
# NDBI
# ============================================================

def calculate_ndbi(
    nir,
    swir,
):
    """
    NDBI:

        (SWIR - NIR) / (SWIR + NIR)

    Sentinel-2:
        B08 = 10 m
        B11 = 20 m

    The caller must align B11 to the B08 reference
    grid before calling this function.
    """

    return normalized_difference(
        swir,
        nir,
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    index,
):
    """
    Calculate robust statistics for a spectral index.

    NaN and infinite values are excluded.
    """

    index = np.asarray(
        index,
        dtype=np.float32,
    )

    valid = index[
        np.isfinite(index)
    ]

    if valid.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "std": np.nan,
            "valid_pixels": 0,
            "valid_fraction": 0.0,
        }

    return {
        "mean": float(
            np.mean(valid)
        ),
        "median": float(
            np.median(valid)
        ),
        "minimum": float(
            np.min(valid)
        ),
        "maximum": float(
            np.max(valid)
        ),
        "std": float(
            np.std(valid)
        ),
        "valid_pixels": int(
            valid.size
        ),
        "valid_fraction": float(
            valid.size
            / index.size
        ),
    }