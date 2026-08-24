"""
Satellite Change Detection
===========================

Compare two Sentinel-2 observations of the
same geographic area.

The module operates on spectral indices
rather than raw RGB imagery.

Supported:
- NDVI
- NDWI
- NDBI
"""

from __future__ import annotations

import numpy as np


# ============================================================
# DIFFERENCE
# ============================================================

def calculate_difference(
    before,
    after,
):
    """
    Calculate after - before.

    Positive values:
        index increased.

    Negative values:
        index decreased.
    """

    before = np.asarray(
        before,
        dtype=np.float32,
    )

    after = np.asarray(
        after,
        dtype=np.float32,
    )

    if before.shape != after.shape:

        raise ValueError(
            "Before and after arrays "
            "must have identical shapes."
        )

    difference = (
        after - before
    )

    difference[
        ~np.isfinite(
            difference
        )
    ] = np.nan

    return difference


# ============================================================
# CHANGE MASK
# ============================================================

def detect_change(
    difference,
    threshold=0.10,
):
    """
    Detect significant change.

    Returns:

        -1 = decrease
         0 = no significant change
        +1 = increase
    """

    difference = np.asarray(
        difference,
        dtype=np.float32,
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

    Default Sentinel-2 pixel:
        10 m × 10 m
    """

    change_map = np.asarray(
        change_map
    )

    pixel_area_km2 = (
        pixel_size_meters ** 2
    ) / 1_000_000.0

    decrease_pixels = np.sum(
        change_map == -1
    )

    increase_pixels = np.sum(
        change_map == 1
    )

    unchanged_pixels = np.sum(
        change_map == 0
    )

    return {

        "decrease_pixels":
            int(decrease_pixels),

        "increase_pixels":
            int(increase_pixels),

        "unchanged_pixels":
            int(unchanged_pixels),

        "decrease_km2":
            float(
                decrease_pixels
                * pixel_area_km2
            ),

        "increase_km2":
            float(
                increase_pixels
                * pixel_area_km2
            ),

        "total_changed_km2":
            float(
                (
                    decrease_pixels
                    + increase_pixels
                )
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
    Calculate relative change:

        (after - before)
        ----------------
        abs(before) + eps

    Useful for detecting relative
    changes in spectral indices.
    """

    before = np.asarray(
        before,
        dtype=np.float32,
    )

    after = np.asarray(
        after,
        dtype=np.float32,
    )

    epsilon = 1e-6

    result = (
        (after - before)
        / (
            np.abs(before)
            + epsilon
        )
    )

    result[
        ~np.isfinite(result)
    ] = np.nan

    return result