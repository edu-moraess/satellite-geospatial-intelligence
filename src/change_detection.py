"""
Satellite Change Detection
===========================

Robust multi-temporal change detection
for Sentinel-2 spectral indices.

Supported:
- NDVI
- NDWI
- NDBI

The module is intentionally strict about
array compatibility. Spatial alignment
should be performed before comparison.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# VALIDATION
# ============================================================

def validate_arrays(
    before,
    after,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert inputs to float32 arrays and validate
    their spatial compatibility.
    """

    before_array = np.asarray(
        before,
        dtype=np.float32,
    )

    after_array = np.asarray(
        after,
        dtype=np.float32,
    )

    if before_array.ndim != 2:
        raise ValueError(
            "Before array must be a 2D raster."
        )

    if after_array.ndim != 2:
        raise ValueError(
            "After array must be a 2D raster."
        )

    if before_array.shape != after_array.shape:
        raise ValueError(
            "Before and after arrays must have "
            "identical shapes. "
            "Align both rasters to the same "
            "spatial reference before comparison."
        )

    return before_array, after_array


# ============================================================
# DIFFERENCE
# ============================================================

def calculate_difference(
    before,
    after,
):
    """
    Calculate:

        after - before

    Positive values:
        index increased.

    Negative values:
        index decreased.

    Both arrays must already represent the
    same spatial grid.
    """

    before_array, after_array = validate_arrays(
        before,
        after,
    )

    difference = (
        after_array
        - before_array
    )

    invalid = (
        ~np.isfinite(before_array)
        | ~np.isfinite(after_array)
        | ~np.isfinite(difference)
    )

    difference[invalid] = np.nan

    return difference


# ============================================================
# CHANGE MASK
# ============================================================

def detect_change(
    difference,
    threshold=0.10,
):
    """
    Detect significant spectral change.

    Returns:

        -1 = significant decrease
         0 = no significant change
        +1 = significant increase
    """

    difference = np.asarray(
        difference,
        dtype=np.float32,
    )

    if difference.ndim != 2:
        raise ValueError(
            "Difference raster must be 2D."
        )

    if not np.isfinite(threshold):
        raise ValueError(
            "Threshold must be finite."
        )

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    result = np.zeros(
        difference.shape,
        dtype=np.int8,
    )

    valid = np.isfinite(
        difference
    )

    result[
        valid
        & (difference < -threshold)
    ] = -1

    result[
        valid
        & (difference > threshold)
    ] = 1

    return result


# ============================================================
# CHANGE STATISTICS
# ============================================================

def calculate_change_statistics(
    change_map,
    pixel_size_meters=10.0,
):
    """
    Calculate changed area.

    Default:
        Sentinel-2 10 m pixel.

    Returns pixel counts and area in km².
    """

    change_map = np.asarray(
        change_map
    )

    if change_map.ndim != 2:
        raise ValueError(
            "Change map must be a 2D raster."
        )

    if pixel_size_meters <= 0:
        raise ValueError(
            "Pixel size must be greater than zero."
        )

    pixel_area_km2 = (
        pixel_size_meters ** 2
    ) / 1_000_000.0

    decrease_pixels = int(
        np.sum(change_map == -1)
    )

    increase_pixels = int(
        np.sum(change_map == 1)
    )

    unchanged_pixels = int(
        np.sum(change_map == 0)
    )

    total_changed_pixels = (
        decrease_pixels
        + increase_pixels
    )

    return {
        "decrease_pixels": decrease_pixels,
        "increase_pixels": increase_pixels,
        "unchanged_pixels": unchanged_pixels,

        "decrease_km2": float(
            decrease_pixels
            * pixel_area_km2
        ),

        "increase_km2": float(
            increase_pixels
            * pixel_area_km2
        ),

        "total_changed_km2": float(
            total_changed_pixels
            * pixel_area_km2
        ),
    }


# ============================================================
# NORMALIZED CHANGE
# ============================================================

def normalized_change(
    before,
    after,
):
    """
    Calculate relative spectral change:

        (after - before)
        ----------------
        abs(before) + epsilon

    Both rasters must have identical shapes.
    """

    before_array, after_array = validate_arrays(
        before,
        after,
    )

    epsilon = 1e-6

    result = (
        after_array
        - before_array
    ) / (
        np.abs(before_array)
        + epsilon
    )

    invalid = (
        ~np.isfinite(before_array)
        | ~np.isfinite(after_array)
        | ~np.isfinite(result)
    )

    result[invalid] = np.nan

    return result