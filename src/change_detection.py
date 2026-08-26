"""
Satellite Change Detection
===========================

Compare two Sentinel-2 observations of the same geographic
area using spectral indices.

Supported:
- NDVI
- NDWI
- NDBI

This module performs mathematical change analysis only.
Raster integrity and spatial compatibility are delegated to
src.raster_validation.
"""

from __future__ import annotations

import numpy as np

from src.raster_validation import (
    validate_raster,
    validate_raster_pair,
)


# ============================================================
# DIFFERENCE
# ============================================================

def calculate_difference(
    before,
    after,
    before_metadata=None,
    after_metadata=None,
):
    """
    Calculate:

        after - before

    Positive values:
        index increased.

    Negative values:
        index decreased.

    When metadata is available, CRS and spatial-grid
    compatibility are validated before subtraction.

    Raises:
        RasterValidationError when rasters are unsafe
        to compare.
    """

    before = np.asarray(
        before,
        dtype=np.float32,
    )

    after = np.asarray(
        after,
        dtype=np.float32,
    )

    validate_raster_pair(
        before,
        after,
        before_metadata,
        after_metadata,
        label_a="before scene",
        label_b="after scene",
    )

    difference = (
        after - before
    ).astype(
        np.float32,
        copy=False,
    )

    difference[
        ~np.isfinite(difference)
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
         0 = unchanged
         1 = increase

    Invalid / NaN pixels are represented as NaN rather than
    being silently classified as unchanged.

    The output is float32 because the map needs to preserve
    the distinction between:
        -1
         0
         1
        NaN
    """

    difference = np.asarray(
        difference,
        dtype=np.float32,
    )

    validate_raster(
        difference,
        label="spectral difference",
    )

    if not np.isfinite(threshold):
        raise ValueError(
            "Change threshold must be finite."
        )

    if threshold < 0:
        raise ValueError(
            "Change threshold cannot be negative."
        )

    result = np.full(
        difference.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(
        difference
    )

    result[
        valid
        & (
            difference
            < -threshold
        )
    ] = -1.0

    result[
        valid
        & (
            np.abs(difference)
            <= threshold
        )
    ] = 0.0

    result[
        valid
        & (
            difference
            > threshold
        )
    ] = 1.0

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

    Invalid pixels are excluded from the denominator and
    from all class areas.

    Default:
        Sentinel-2 10 m grid.
    """

    change_map = np.asarray(
        change_map,
        dtype=np.float32,
    )

    validate_raster(
        change_map,
        label="change map",
    )

    if (
        not np.isfinite(
            pixel_size_meters
        )
        or pixel_size_meters <= 0
    ):
        raise ValueError(
            "pixel_size_meters must be "
            "a positive finite number."
        )

    pixel_area_km2 = (
        pixel_size_meters ** 2
    ) / 1_000_000.0

    valid = np.isfinite(
        change_map
    )

    decrease_pixels = int(
        np.sum(
            valid
            & (change_map == -1)
        )
    )

    increase_pixels = int(
        np.sum(
            valid
            & (change_map == 1)
        )
    )

    unchanged_pixels = int(
        np.sum(
            valid
            & (change_map == 0)
        )
    )

    valid_pixels = int(
        np.sum(valid)
    )

    total_pixels = int(
        change_map.size
    )

    changed_pixels = (
        decrease_pixels
        + increase_pixels
    )

    return {
        "decrease_pixels":
            decrease_pixels,

        "increase_pixels":
            increase_pixels,

        "unchanged_pixels":
            unchanged_pixels,

        "valid_pixels":
            valid_pixels,

        "invalid_pixels":
            total_pixels
            - valid_pixels,

        "changed_pixels":
            changed_pixels,

        "valid_fraction":
            float(
                valid_pixels
                / total_pixels
            )
            if total_pixels
            else 0.0,

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

        "unchanged_km2":
            float(
                unchanged_pixels
                * pixel_area_km2
            ),

        "total_changed_km2":
            float(
                changed_pixels
                * pixel_area_km2
            ),

        "analyzed_area_km2":
            float(
                valid_pixels
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
        abs(before) + epsilon

    Pixels where either observation is invalid are returned
    as NaN.
    """

    before = np.asarray(
        before,
        dtype=np.float32,
    )

    after = np.asarray(
        after,
        dtype=np.float32,
    )

    validate_raster_pair(
        before,
        after,
        label_a="before scene",
        label_b="after scene",
    )

    epsilon = np.float32(
        1e-6
    )

    result = np.full(
        before.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(before)
        & np.isfinite(after)
    )

    denominator = (
        np.abs(before)
        + epsilon
    )

    safe = (
        valid
        & np.isfinite(denominator)
        & (
            denominator
            > epsilon
        )
    )

    result[safe] = (
        (
            after[safe]
            - before[safe]
        )
        / denominator[safe]
    )

    result[
        ~np.isfinite(result)
    ] = np.nan

    return result