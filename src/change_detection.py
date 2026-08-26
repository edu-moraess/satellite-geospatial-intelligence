"""
Change Detection Engine
=======================

Responsible for:

    - validating Before / After rasters;
    - calculating pixel-wise spectral differences;
    - detecting significant increases/decreases;
    - calculating spatial change statistics.

Core rule:

    NO pixel-wise operation is executed before the input
    rasters pass raster_validation.validate_raster_pair().

Difference convention:

    difference = AFTER - BEFORE

Therefore:

    +difference -> increase
    -difference -> decrease
     0          -> no change

The module never silently crops, pads, broadcasts or invents
pixels.

If the Before and After rasters do not share the same spatial
grid, the operation is rejected with RasterValidationError.
"""

from __future__ import annotations


import numpy as np

from src.raster_validation import (
    RasterValidationError,
    validate_raster,
    validate_raster_pair,
)

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_THRESHOLD = 0.10


# ============================================================
# INTERNAL VALIDATION
# ============================================================

def _common_valid_mask(array_a, array_b):
    """Return boolean mask of pixels where both arrays are finite."""
    return np.isfinite(array_a) & np.isfinite(array_b)


def _validate_difference_inputs(
    before,
    after,
    before_metadata=None,
    after_metadata=None,
):
    """
    Validate Before and After arrays before any arithmetic.

    Returns:
        validation diagnostics.
    """
    # validate_raster_pair only accepts require_same_dtype
    return validate_raster_pair(
        before,
        after,
        metadata_a=before_metadata,
        metadata_b=after_metadata,
        label_a="Before",
        label_b="After",
        require_same_dtype=False,
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
    Calculate spectral difference:

        difference = after - before

    Before performing subtraction, both rasters are validated
    for:

        - 2D structure;
        - non-empty data;
        - finite values;
        - equal shape;
        - compatible CRS;
        - identical spatial transform;
        - overlapping valid pixels.

    Parameters:
        before:
            Before raster/index.

        after:
            After raster/index.

        before_metadata:
            Spatial metadata for Before.

        after_metadata:
            Spatial metadata for After.

    Returns:
        float32 difference raster.

    Raises:
        RasterValidationError
            If the rasters are not spatially compatible.
    """
    _validate_difference_inputs(
        before,
        after,
        before_metadata,
        after_metadata,
    )

    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)

    # Check shape compatibility (already done by validate_raster_pair,
    # but we double-check)
    if before.shape != after.shape:
        raise RasterValidationError(
            f"Shape mismatch: Before {before.shape}, After {after.shape}"
        )

    valid_mask = _common_valid_mask(before, after)

    # Start with NaN everywhere so invalid pixels are never
    # interpreted as actual zero change.
    difference = np.full(before.shape, np.nan, dtype=np.float32)
    difference[valid_mask] = after[valid_mask] - before[valid_mask]

    return difference


# ============================================================
# CHANGE CLASSIFICATION
# ============================================================

def detect_change(
    difference,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Convert a continuous difference raster into a categorical
    change map.

    Classification:

        -1 = significant decrease
         0 = no significant change
        +1 = significant increase

    Threshold is symmetric:

        difference <= -threshold -> decrease
        difference >= +threshold -> increase

    Invalid pixels remain NaN.

    This is important because NoData must not be converted into
    "unchanged".
    """
    validate_raster(difference, label="spectral difference")

    if not np.isfinite(threshold):
        raise ValueError("Change threshold must be finite.")

    if threshold < 0:
        raise ValueError("Change threshold must be >= 0.")

    difference = np.asarray(difference, dtype=np.float32)
    change_map = np.full(difference.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(difference)
    decrease = valid & (difference <= -threshold)
    increase = valid & (difference >= threshold)
    unchanged = valid & ~decrease & ~increase

    change_map[decrease] = -1.0
    change_map[unchanged] = 0.0
    change_map[increase] = 1.0

    return change_map


# ============================================================
# CHANGE MASK
# ============================================================

def get_change_mask(change_map):
    """
    Return a boolean mask identifying significant change.

    True:
        increase or decrease.

    False:
        unchanged or invalid.
    """
    validate_raster(change_map, label="change map")
    change_map = np.asarray(change_map)
    return np.isfinite(change_map) & (change_map != 0)


# ============================================================
# VALIDITY MASK
# ============================================================

def get_valid_mask(change_map):
    """
    Return pixels that contain a valid change classification.
    """
    change_map = np.asarray(change_map)
    if change_map.ndim != 2:
        raise RasterValidationError("Change map must be a 2D array.")
    return np.isfinite(change_map)


# ============================================================
# CHANGE STATISTICS
# ============================================================

def calculate_change_statistics(
    change_map,
    pixel_size_meters: float = 10.0,
):
    """
    Calculate spatial change statistics.

    Parameters:
        change_map:
            Categorical map:
                -1 decrease
                 0 unchanged
                +1 increase

        pixel_size_meters:
            Pixel side length in meters.

    Returns:
        Dictionary containing:
            decrease_pixels
            increase_pixels
            unchanged_pixels
            valid_pixels
            total_changed_pixels
            decrease_km2
            increase_km2
            unchanged_km2
            total_changed_km2
            valid_area_km2
            changed_fraction
            decrease_fraction
            increase_fraction
    """
    validate_raster(change_map, label="change map")

    if not np.isfinite(pixel_size_meters):
        raise ValueError("pixel_size_meters must be finite.")

    if pixel_size_meters <= 0:
        raise ValueError("pixel_size_meters must be greater than zero.")

    change_map = np.asarray(change_map, dtype=np.float32)

    valid = np.isfinite(change_map)
    decrease = valid & (change_map < 0)
    increase = valid & (change_map > 0)
    unchanged = valid & (change_map == 0)

    decrease_pixels = int(np.count_nonzero(decrease))
    increase_pixels = int(np.count_nonzero(increase))
    unchanged_pixels = int(np.count_nonzero(unchanged))
    valid_pixels = int(np.count_nonzero(valid))
    total_changed_pixels = decrease_pixels + increase_pixels

    pixel_area_km2 = (float(pixel_size_meters) ** 2) / 1_000_000.0

    decrease_km2 = decrease_pixels * pixel_area_km2
    increase_km2 = increase_pixels * pixel_area_km2
    unchanged_km2 = unchanged_pixels * pixel_area_km2
    total_changed_km2 = total_changed_pixels * pixel_area_km2
    valid_area_km2 = valid_pixels * pixel_area_km2

    if valid_pixels > 0:
        changed_fraction = total_changed_pixels / valid_pixels
        decrease_fraction = decrease_pixels / valid_pixels
        increase_fraction = increase_pixels / valid_pixels
    else:
        changed_fraction = 0.0
        decrease_fraction = 0.0
        increase_fraction = 0.0

    return {
        "decrease_pixels": decrease_pixels,
        "increase_pixels": increase_pixels,
        "unchanged_pixels": unchanged_pixels,
        "valid_pixels": valid_pixels,
        "total_changed_pixels": total_changed_pixels,
        "decrease_km2": float(decrease_km2),
        "increase_km2": float(increase_km2),
        "unchanged_km2": float(unchanged_km2),
        "total_changed_km2": float(total_changed_km2),
        "valid_area_km2": float(valid_area_km2),
        "changed_fraction": float(changed_fraction),
        "decrease_fraction": float(decrease_fraction),
        "increase_fraction": float(increase_fraction),
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize_change(
    change_map,
    pixel_size_meters: float = 10.0,
):
    """
    Convenience wrapper returning the main operational
    statistics for UI presentation.
    """
    stats = calculate_change_statistics(
        change_map,
        pixel_size_meters=pixel_size_meters,
    )

    return {
        "changed_km2": stats["total_changed_km2"],
        "decrease_km2": stats["decrease_km2"],
        "increase_km2": stats["increase_km2"],
        "changed_fraction": stats["changed_fraction"],
        "valid_area_km2": stats["valid_area_km2"],
    }