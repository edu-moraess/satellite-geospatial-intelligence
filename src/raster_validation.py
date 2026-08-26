# Raster Alignment / Validation Layer

"""
Single source of truth for deciding whether one raster or
two rasters are safe to use in mathematical operations.

This module NEVER invents data. It only inspects arrays and
metadata already present in the pipeline.
"""

from __future__ import annotations

import numpy as np


class RasterValidationError(ValueError):
    """Raised when raster data is unsafe for downstream use."""


def _metadata_get(metadata, key):
    """Safely retrieve a metadata field."""
    if metadata is None:
        return None
    return metadata.get(key)


def _transforms_close(transform_a, transform_b, tolerance=1e-6):
    """Compare affine transforms with numerical tolerance."""
    if transform_a is None or transform_b is None:
        return transform_a == transform_b

    try:
        values_a = tuple(transform_a)[:6]
        values_b = tuple(transform_b)[:6]
    except TypeError:
        return transform_a == transform_b

    if len(values_a) != len(values_b):
        return False

    return all(
        abs(float(a) - float(b)) <= tolerance
        for a, b in zip(values_a, values_b)
    )


def validate_raster(
    array,
    metadata=None,
    *,
    label="raster",
    min_valid_fraction=0.0,
):
    """
    Validate a single raster array.

    Returns diagnostic information when valid.
    Raises RasterValidationError when invalid.
    """
    if array is None:
        raise RasterValidationError(
            f"{label}: raster is missing (None). "
            "The raster was not loaded correctly."
        )

    try:
        array = np.asarray(array)
    except Exception as error:
        raise RasterValidationError(
            f"{label}: could not convert raster to a NumPy array."
        ) from error

    if array.size == 0:
        raise RasterValidationError(
            f"{label}: raster is empty (size 0)."
        )

    if array.ndim != 2:
        raise RasterValidationError(
            f"{label}: expected a 2D single-band array, "
            f"got shape {array.shape} (ndim={array.ndim})."
        )

    try:
        finite_mask = np.isfinite(array)
    except TypeError as error:
        raise RasterValidationError(
            f"{label}: raster dtype {array.dtype} "
            "cannot be validated as numeric data."
        ) from error

    valid_fraction = float(
        np.mean(finite_mask)
    )

    if valid_fraction == 0.0:
        raise RasterValidationError(
            f"{label}: raster has no finite values. "
            "The scene may be fully nodata or corrupted."
        )

    if valid_fraction < min_valid_fraction:
        raise RasterValidationError(
            f"{label}: only {valid_fraction:.1%} of pixels "
            f"are valid, below the required "
            f"{min_valid_fraction:.1%}."
        )

    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "valid_fraction": valid_fraction,
    }


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
    Validate two rasters before a pixel-wise operation.

    Checks:
        - both rasters are valid;
        - identical shape;
        - compatible CRS;
        - compatible spatial transform;
        - optional dtype equality;
        - at least one overlapping finite pixel.
    """
    info_a = validate_raster(
        array_a,
        metadata_a,
        label=label_a,
    )

    info_b = validate_raster(
        array_b,
        metadata_b,
        label=label_b,
    )

    array_a = np.asarray(array_a)
    array_b = np.asarray(array_b)

    if array_a.shape != array_b.shape:
        raise RasterValidationError(
            "Raster shape mismatch.\n\n"
            f"{label_a}: {array_a.shape}\n"
            f"{label_b}: {array_b.shape}\n\n"
            "Align the second raster to the reference grid "
            "before performing a pixel-wise operation."
        )

    crs_a = _metadata_get(
        metadata_a,
        "crs",
    )

    crs_b = _metadata_get(
        metadata_b,
        "crs",
    )

    if (
        crs_a is not None
        and crs_b is not None
        and crs_a != crs_b
    ):
        raise RasterValidationError(
            "CRS mismatch between rasters.\n\n"
            f"{label_a}: {crs_a}\n"
            f"{label_b}: {crs_b}\n\n"
            "Both rasters must use the same CRS before "
            "a pixel-wise comparison."
        )

    transform_a = _metadata_get(
        metadata_a,
        "transform",
    )

    transform_b = _metadata_get(
        metadata_b,
        "transform",
    )

    if (
        transform_a is not None
        and transform_b is not None
        and not _transforms_close(
            transform_a,
            transform_b,
        )
    ):
        raise RasterValidationError(
            "Spatial grid mismatch between rasters.\n\n"
            f"{label_a}: {transform_a}\n"
            f"{label_b}: {transform_b}\n\n"
            "The rasters do not share the same pixel grid. "
            "Align them to a common reference grid first."
        )

    if (
        require_same_dtype
        and array_a.dtype != array_b.dtype
    ):
        raise RasterValidationError(
            "Dtype mismatch between rasters.\n\n"
            f"{label_a}: {array_a.dtype}\n"
            f"{label_b}: {array_b.dtype}"
        )

    overlap = (
        np.isfinite(array_a)
        & np.isfinite(array_b)
    )

    overlap_fraction = float(
        np.mean(overlap)
    )

    if not np.any(overlap):
        raise RasterValidationError(
            f"{label_a} and {label_b} have no overlapping "
            "valid pixels. The mathematical result would "
            "contain no valid pixels."
        )

    return {
        label_a: info_a,
        label_b: info_b,
        "shape": array_a.shape,
        "overlap_fraction": overlap_fraction,
    }