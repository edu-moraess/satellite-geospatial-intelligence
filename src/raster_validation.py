"""
Raster Alignment / Validation Layer
=====================================

Single source of truth for deciding whether two rasters
(or a single raster) are safe to use in a mathematical
operation (spectral index, change detection difference,
band stacking, etc).

Every code path that does `band_a + band_b`, `after - before`,
or any other pixel-wise operation between two arrays should
go through `validate_raster_pair()` first.

This module NEVER invents data. It only inspects what is
already present in the array / metadata and raises a typed,
human-readable error when something is unsafe.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# ERROR TYPES
# ============================================================

class RasterValidationError(ValueError):
    """
    Raised when a raster (or a pair of rasters) is not safe
    to use in a downstream operation.

    Kept as a distinct type (instead of a bare ValueError) so
    the UI layer can catch it specifically and show a clear,
    actionable message instead of a raw traceback.
    """


# ============================================================
# HELPERS
# ============================================================

def _metadata_get(metadata, key):
    """
    Defensive accessor: metadata dicts coming from different
    parts of the pipeline are not guaranteed to have every key.
    """

    if metadata is None:
        return None

    return metadata.get(key)


def _transforms_close(transform_a, transform_b, tolerance=1e-6):
    """
    Compare two affine transforms with a small numerical
    tolerance instead of strict equality, since transforms
    read back from different GeoTIFFs (or reprojected once)
    can differ by floating point noise even when they
    represent the same grid.
    """

    if transform_a is None or transform_b is None:
        return transform_a == transform_b

    values_a = tuple(transform_a)[:6]
    values_b = tuple(transform_b)[:6]

    return all(
        abs(a - b) <= tolerance
        for a, b in zip(values_a, values_b)
    )


# ============================================================
# SINGLE RASTER VALIDATION
# ============================================================

def validate_raster(array, metadata=None, *, label="raster", min_valid_fraction=0.0):
    """
    Validate a single raster array before using it for
    anything (display, index calculation, detection, etc).

    Checks:
        - array is a numpy array
        - array is not empty
        - array is 2D (single band)
        - array contains at least one finite value
        - optionally: at least `min_valid_fraction` of pixels
          are finite (useful to reject scenes that are almost
          entirely nodata/cloud-masked)

    Raises:
        RasterValidationError
    """

    if array is None:
        raise RasterValidationError(
            f"{label}: raster is missing (None). "
            "The band was not loaded correctly."
        )

    array = np.asarray(array)

    if array.size == 0:
        raise RasterValidationError(
            f"{label}: raster is empty (size 0)."
        )

    if array.ndim != 2:
        raise RasterValidationError(
            f"{label}: expected a 2D single-band array, "
            f"got shape {array.shape} (ndim={array.ndim})."
        )

    finite_mask = np.isfinite(array)
    valid_fraction = float(np.mean(finite_mask)) if array.size else 0.0

    if valid_fraction == 0.0:
        raise RasterValidationError(
            f"{label}: raster has no finite values. "
            "The scene may be fully nodata, or the download "
            "may be incomplete/corrupted."
        )

    if valid_fraction < min_valid_fraction:
        raise RasterValidationError(
            f"{label}: only {valid_fraction:.1%} of pixels are "
            f"valid, below the required {min_valid_fraction:.1%}. "
            "Try a different scene or a lower cloud-cover filter."
        )

    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "valid_fraction": valid_fraction,
    }


# ============================================================
# RASTER PAIR VALIDATION
# ============================================================

def validate_raster_pair(
    array_a,
    array_b,
    metadata_a=None,
    metadata_b=None,
    *,
    label_a="band A",
    label_b="band B",
    require_same_dtype=False,
):
    """
    Validate that two rasters are safe to combine in a
    pixel-wise mathematical operation.

    Checks, in order:
        1. Both arrays pass `validate_raster()` individually.
        2. Same shape.
        3. Same CRS (if metadata provided).
        4. Same transform, i.e. same pixel grid / origin /
           resolution (if metadata provided).
        5. Same dtype (only if `require_same_dtype=True` -
           usually not required since both get cast to
           float32 before the operation).
        6. At least one pixel is finite in both arrays at the
           same location (otherwise the operation would
           produce an all-NaN result silently).

    Raises:
        RasterValidationError with a specific, actionable
        message identifying exactly which check failed.

    Returns:
        A small diagnostic dict, useful for logging /
        surfacing in the UI ("Raster Alignment: OK").
    """

    info_a = validate_raster(array_a, metadata_a, label=label_a)
    info_b = validate_raster(array_b, metadata_b, label=label_b)

    array_a = np.asarray(array_a)
    array_b = np.asarray(array_b)

    # ---- 2. Shape ----
    if array_a.shape != array_b.shape:
        raise RasterValidationError(
            "Raster shape mismatch.\n\n"
            f"{label_a}: {array_a.shape}\n"
            f"{label_b}: {array_b.shape}\n\n"
            "The lower-resolution band must be resampled to "
            "the reference grid before this operation "
            "(see geospatial.align_band_to_reference)."
        )

    # ---- 3. CRS ----
    crs_a = _metadata_get(metadata_a, "crs")
    crs_b = _metadata_get(metadata_b, "crs")

    if crs_a is not None and crs_b is not None and crs_a != crs_b:
        raise RasterValidationError(
            "CRS mismatch between rasters.\n\n"
            f"{label_a}: {crs_a}\n"
            f"{label_b}: {crs_b}\n\n"
            "Both rasters must be reprojected to a common CRS "
            "before comparison."
        )

    # ---- 4. Transform (pixel grid) ----
    transform_a = _metadata_get(metadata_a, "transform")
    transform_b = _metadata_get(metadata_b, "transform")

    if (
        transform_a is not None
        and transform_b is not None
        and not _transforms_close(transform_a, transform_b)
    ):
        raise RasterValidationError(
            "Spatial grid (transform) mismatch between rasters.\n\n"
            f"{label_a}: {transform_a}\n"
            f"{label_b}: {transform_b}\n\n"
            "The rasters do not share the same pixel grid "
            "(origin/resolution/rotation). Align them with "
            "geospatial.align_band_to_reference before combining."
        )

    # ---- 5. Dtype (optional) ----
    if require_same_dtype and array_a.dtype != array_b.dtype:
        raise RasterValidationError(
            "Dtype mismatch between rasters.\n\n"
            f"{label_a}: {array_a.dtype}\n"
            f"{label_b}: {array_b.dtype}"
        )

    # ---- 6. Overlapping valid pixels ----
    overlap = np.isfinite(array_a) & np.isfinite(array_b)

    if not np.any(overlap):
        raise RasterValidationError(
            f"{label_a} and {label_b} have no overlapping "
            "valid pixels. The result would be entirely NaN."
        )

    return {
        label_a: info_a,
        label_b: info_b,
        "shape": array_a.shape,
        "overlap_fraction": float(np.mean(overlap)),
    }
