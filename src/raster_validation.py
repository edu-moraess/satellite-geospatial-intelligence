"src/raster_validation.py"

"""
Raster Alignment / Validation Layer
===================================

Single source of truth for deciding whether one or two rasters
are safe to use in mathematical / pixel-wise operations.

This module NEVER invents data.

It only:
    - inspects arrays;
    - validates dimensions;
    - validates finite values;
    - validates CRS;
    - validates spatial transforms;
    - validates overlapping valid pixels.

Any downstream operation such as:

    after - before
    band_a + band_b
    band_a / band_b
    NDVI
    NDWI
    NDBI
    band stacking

should validate its input rasters through this module first.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ============================================================
# ERROR TYPES
# ============================================================

class RasterValidationError(ValueError):
    """
    Raised when a raster or raster pair is unsafe for
    downstream mathematical processing.

    This distinct exception allows the UI layer to show a
    clean, actionable message instead of a generic traceback.
    """


# ============================================================
# HELPERS
# ============================================================

def _metadata_get(
    metadata: dict | None,
    key: str,
) -> Any:
    """
    Safely retrieve a metadata field.
    """

    if metadata is None:
        return None

    return metadata.get(key)


def _transform_values(transform) -> tuple[float, ...] | None:
    """
    Convert an affine-like transform into its first six
    numerical coefficients.

    Returns None when the transform cannot be interpreted.
    """

    if transform is None:
        return None

    try:
        values = tuple(transform)[:6]

        if len(values) != 6:
            return None

        return tuple(float(value) for value in values)

    except (TypeError, ValueError):
        return None


def _transforms_close(
    transform_a,
    transform_b,
    tolerance: float = 1e-6,
) -> bool:
    """
    Compare two affine transforms using numerical tolerance.

    Exact equality is intentionally avoided because transforms
    reconstructed by rasterio/reprojection can differ by tiny
    floating-point errors while representing the same grid.
    """

    values_a = _transform_values(transform_a)
    values_b = _transform_values(transform_b)

    if values_a is None or values_b is None:
        return transform_a == transform_b

    return bool(
        np.allclose(
            values_a,
            values_b,
            rtol=0.0,
            atol=tolerance,
        )
    )


def transforms_close(
    transform_a,
    transform_b,
    tolerance: float = 1e-6,
) -> bool:
    """
    Public wrapper around transform comparison.

    Useful for other modules that need to determine whether
    two rasters share the same spatial grid.
    """

    return _transforms_close(
        transform_a,
        transform_b,
        tolerance=tolerance,
    )


def _crs_equal(
    crs_a,
    crs_b,
) -> bool:
    """
    Compare CRS objects safely.

    rasterio CRS objects generally support equality, but this
    helper also handles string representations.
    """

    if crs_a is None or crs_b is None:
        return crs_a == crs_b

    try:
        return bool(crs_a == crs_b)
    except Exception:
        return str(crs_a) == str(crs_b)


# ============================================================
# SINGLE RASTER VALIDATION
# ============================================================

def validate_raster(
    array,
    metadata: dict | None = None,
    *,
    label: str = "raster",
    min_valid_fraction: float = 0.0,
) -> dict:
    """
    Validate a single raster before downstream processing.

    Checks:
        1. Array exists.
        2. Array is non-empty.
        3. Array is 2D.
        4. Array contains finite values.
        5. Optional minimum valid-pixel fraction.

    Returns:
        Diagnostic dictionary.
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

    valid_fraction = (
        float(np.mean(finite_mask))
        if array.size
        else 0.0
    )

    if valid_fraction <= 0.0:
        raise RasterValidationError(
            f"{label}: raster has no finite values. "
            "The scene may be fully nodata, or the download "
            "may be incomplete/corrupted."
        )

    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError(
            "min_valid_fraction must be between 0.0 and 1.0."
        )

    if valid_fraction < min_valid_fraction:
        raise RasterValidationError(
            f"{label}: only {valid_fraction:.1%} of pixels are "
            f"valid, below the required "
            f"{min_valid_fraction:.1%}. "
            "Try a different scene or a lower cloud-cover "
            "filter."
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
    metadata_a: dict | None = None,
    metadata_b: dict | None = None,
    *,
    label_a: str = "raster A",
    label_b: str = "raster B",
    require_same_dtype: bool = False,
    require_same_crs: bool = True,
    require_same_transform: bool = True,
    transform_tolerance: float = 1e-6,
    require_overlap: bool = True,
) -> dict:
    """
    Validate that two rasters are safe for a pixel-wise
    operation.

    Validation order:

        1. Individual raster validity.
        2. Shape.
        3. CRS.
        4. Spatial transform / pixel grid.
        5. Optional dtype.
        6. Overlapping finite pixels.

    IMPORTANT:
        This function does not resample or reproject data.

        If validation fails, the caller must explicitly align
        the rasters first.
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

    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    if array_a.shape != array_b.shape:
        raise RasterValidationError(
            "Raster shape mismatch.\n\n"
            f"{label_a}: {array_a.shape}\n"
            f"{label_b}: {array_b.shape}\n\n"
            "The rasters must be resampled to the same "
            "reference grid before the pixel-wise operation."
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    crs_a = _metadata_get(
        metadata_a,
        "crs",
    )

    crs_b = _metadata_get(
        metadata_b,
        "crs",
    )

    if (
        require_same_crs
        and crs_a is not None
        and crs_b is not None
        and not _crs_equal(crs_a, crs_b)
    ):
        raise RasterValidationError(
            "CRS mismatch between rasters.\n\n"
            f"{label_a}: {crs_a}\n"
            f"{label_b}: {crs_b}\n\n"
            "Both rasters must be reprojected to a common "
            "coordinate reference system before comparison."
        )

    # --------------------------------------------------------
    # SPATIAL GRID / TRANSFORM
    # --------------------------------------------------------

    transform_a = _metadata_get(
        metadata_a,
        "transform",
    )

    transform_b = _metadata_get(
        metadata_b,
        "transform",
    )

    if (
        require_same_transform
        and transform_a is not None
        and transform_b is not None
        and not _transforms_close(
            transform_a,
            transform_b,
            tolerance=transform_tolerance,
        )
    ):
        raise RasterValidationError(
            "Spatial grid (transform) mismatch between rasters.\n\n"
            f"{label_a}: {transform_a}\n"
            f"{label_b}: {transform_b}\n\n"
            "The rasters do not share the same pixel grid "
            "(origin, resolution or rotation).\n\n"
            "Align the secondary raster to the reference "
            "grid before performing the operation."
        )

    # --------------------------------------------------------
    # DTYPE
    # --------------------------------------------------------

    if (
        require_same_dtype
        and array_a.dtype != array_b.dtype
    ):
        raise RasterValidationError(
            "Dtype mismatch between rasters.\n\n"
            f"{label_a}: {array_a.dtype}\n"
            f"{label_b}: {array_b.dtype}"
        )

    # --------------------------------------------------------
    # OVERLAPPING VALID PIXELS
    # --------------------------------------------------------

    overlap = (
        np.isfinite(array_a)
        & np.isfinite(array_b)
    )

    overlap_fraction = float(
        np.mean(overlap)
    )

    if require_overlap and not np.any(overlap):
        raise RasterValidationError(
            f"{label_a} and {label_b} have no overlapping "
            "valid pixels. The result would be entirely NaN."
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        label_a: info_a,
        label_b: info_b,
        "shape": array_a.shape,
        "overlap_fraction": overlap_fraction,
        "crs_match": (
            True
            if crs_a is None or crs_b is None
            else _crs_equal(crs_a, crs_b)
        ),
        "transform_match": (
            True
            if transform_a is None or transform_b is None
            else _transforms_close(
                transform_a,
                transform_b,
                tolerance=transform_tolerance,
            )
        ),
        "safe": True,
    }


# ============================================================
# SAME GRID CHECK
# ============================================================

def rasters_share_grid(
    array_a,
    array_b,
    metadata_a: dict | None = None,
    metadata_b: dict | None = None,
    *,
    transform_tolerance: float = 1e-6,
) -> bool:
    """
    Return True only when two rasters have the same shape,
    CRS and spatial transform.

    This is a non-raising convenience function.
    """

    try:
        validate_raster_pair(
            array_a,
            array_b,
            metadata_a,
            metadata_b,
            transform_tolerance=transform_tolerance,
        )
        return True
    except RasterValidationError:
        return False


# ============================================================
# VALIDITY MASK
# ============================================================

def common_valid_mask(
    array_a,
    array_b,
) -> np.ndarray:
    """
    Return the finite-pixel intersection of two already
    validated/aligned rasters.

    This function does not perform alignment.
    """

    array_a = np.asarray(array_a)
    array_b = np.asarray(array_b)

    if array_a.shape != array_b.shape:
        raise RasterValidationError(
            "Cannot create a common validity mask from rasters "
            "with different shapes."
        )

    return (
        np.isfinite(array_a)
        & np.isfinite(array_b)
    )