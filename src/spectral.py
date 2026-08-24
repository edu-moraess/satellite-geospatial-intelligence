"""
Spectral index calculations.

Supported indices:

NDVI -> vegetation
NDWI -> water
NDBI -> built-up areas
"""

from __future__ import annotations

import numpy as np


# ============================================================
# SAFE FLOAT ARRAY
# ============================================================

def _to_float_array(
    array,
):
    """
    Convert an input array to float32.
    """

    return np.asarray(
        array,
        dtype=np.float32,
    )


# ============================================================
# ALIGN ARRAYS
# ============================================================

def _validate_same_shape(
    band_a,
    band_b,
):
    """
    Validate that two arrays have identical dimensions.
    """

    if band_a.shape != band_b.shape:

        raise ValueError(
            "Spectral bands have different "
            "spatial dimensions.\n\n"
            f"Band A: {band_a.shape}\n"
            f"Band B: {band_b.shape}\n\n"
            "The lower-resolution band must "
            "be resampled before calculating "
            "the spectral index."
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
    """

    band_a = _to_float_array(
        band_a
    )

    band_b = _to_float_array(
        band_b
    )

    _validate_same_shape(
        band_a,
        band_b,
    )

    numerator = (
        band_a - band_b
    )

    denominator = (
        band_a + band_b
    )

    result = np.full_like(
        denominator,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(band_a)
        & np.isfinite(band_b)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-10)
    )

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

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

    Important:
    Sentinel-2 B11 is normally 20 m while
    B08 is 10 m.

    The caller must align/resample the
    arrays before passing them here.
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
    }