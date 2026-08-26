"""
Geospatial utilities for Satellite Geospatial Intelligence.

Responsibilities:
- Read GeoTIFF bands
- Validate raster structure
- Align/resample bands to a reference grid
- Preserve spatial metadata
- Normalize imagery for visualization
- Stack already-aligned bands

This module does not perform scientific interpretation.
It is responsible only for raster I/O, spatial alignment,
and safe geospatial transformations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.raster_validation import (
    RasterValidationError,
    validate_raster,
    validate_raster_pair,
)


# ============================================================
# CONSTANTS
# ============================================================

TRANSFORM_TOLERANCE = 1e-6


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _transforms_close(
    transform_a,
    transform_b,
    tolerance: float = TRANSFORM_TOLERANCE,
) -> bool:
    """
    Compare two raster transforms using numerical tolerance.

    Exact equality is avoided because affine transforms can
    contain very small floating-point differences while still
    representing the same spatial grid.
    """

    if transform_a is None or transform_b is None:
        return transform_a == transform_b

    values_a = tuple(transform_a)[:6]
    values_b = tuple(transform_b)[:6]

    if len(values_a) != len(values_b):
        return False

    return all(
        abs(float(a) - float(b)) <= tolerance
        for a, b in zip(values_a, values_b)
    )


def _metadata_from_reference(
    reference_metadata,
    *,
    shape,
    transform,
):
    """
    Build metadata describing an aligned raster.

    The returned metadata preserves the reference CRS/grid
    while updating dimensions and transform when necessary.
    """

    metadata = dict(
        reference_metadata or {}
    )

    metadata["height"] = int(shape[0])
    metadata["width"] = int(shape[1])
    metadata["transform"] = transform

    return metadata


# ============================================================
# READ BAND
# ============================================================

def read_band(path):
    """
    Read a single-band GeoTIFF.

    Returns:
        tuple:
            data: float32 numpy array
            metadata: spatial metadata dictionary

    Raises:
        FileNotFoundError
        RasterValidationError
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Band file not found: {path}"
        )

    if not path.is_file():
        raise RasterValidationError(
            f"Band path is not a file: {path}"
        )

    try:
        with rasterio.open(path) as src:

            if src.count < 1:
                raise RasterValidationError(
                    f"GeoTIFF contains no raster bands: {path}"
                )

            data = src.read(1).astype(
                np.float32,
                copy=False,
            )

            metadata = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "bounds": src.bounds,
                "dtype": str(src.dtypes[0]),
                "count": src.count,
                "profile": src.profile.copy(),
            }

    except RasterValidationError:
        raise

    except Exception as error:
        raise RasterValidationError(
            f"Unable to read raster band: {path}\n\n"
            f"Error: {error}"
        ) from error

    validate_raster(
        data,
        metadata,
        label=str(path),
    )

    return data, metadata


# ============================================================
# RESAMPLE TO REFERENCE
# ============================================================

def resample_to_reference(
    source_array,
    source_metadata,
    reference_metadata,
    resampling=Resampling.bilinear,
):
    """
    Resample a raster to the exact spatial grid of a
    reference raster.

    This function performs:
        - reprojection when CRS differs;
        - resampling when resolution differs;
        - translation when origin differs;
        - shape normalization to the reference dimensions.

    The output always uses:
        reference width
        reference height
        reference transform
        reference CRS

    Parameters
    ----------
    source_array:
        2D source raster.

    source_metadata:
        Metadata describing source raster.

    reference_metadata:
        Metadata describing target/reference raster.

    resampling:
        rasterio Resampling method.

    Returns
    -------
    numpy.ndarray
        Float32 raster aligned to the reference grid.
    """

    validate_raster(
        source_array,
        source_metadata,
        label="source raster",
    )

    if source_metadata is None:
        raise RasterValidationError(
            "Source raster metadata is required "
            "for spatial alignment."
        )

    if reference_metadata is None:
        raise RasterValidationError(
            "Reference raster metadata is required "
            "for spatial alignment."
        )

    required_source = [
        "transform",
        "crs",
    ]

    required_reference = [
        "transform",
        "crs",
        "width",
        "height",
    ]

    missing_source = [
        key
        for key in required_source
        if source_metadata.get(key) is None
    ]

    missing_reference = [
        key
        for key in required_reference
        if reference_metadata.get(key) is None
    ]

    if missing_source:
        raise RasterValidationError(
            "Source raster metadata is incomplete.\n\n"
            f"Missing: {', '.join(missing_source)}"
        )

    if missing_reference:
        raise RasterValidationError(
            "Reference raster metadata is incomplete.\n\n"
            f"Missing: {', '.join(missing_reference)}"
        )

    reference_shape = (
        int(reference_metadata["height"]),
        int(reference_metadata["width"]),
    )

    destination = np.full(
        reference_shape,
        np.nan,
        dtype=np.float32,
    )

    source_nodata = source_metadata.get(
        "nodata"
    )

    destination_nodata = np.nan

    try:
        reproject(
            source=np.asarray(
                source_array,
                dtype=np.float32,
            ),
            destination=destination,
            src_transform=source_metadata["transform"],
            src_crs=source_metadata["crs"],
            dst_transform=reference_metadata["transform"],
            dst_crs=reference_metadata["crs"],
            src_nodata=source_nodata,
            dst_nodata=destination_nodata,
            resampling=resampling,
            init_dest_nodata=True,
        )

    except Exception as error:
        raise RasterValidationError(
            "Raster reprojection/resampling failed.\n\n"
            f"Source CRS: {source_metadata.get('crs')}\n"
            f"Target CRS: {reference_metadata.get('crs')}\n"
            f"Error: {error}"
        ) from error

    destination = np.asarray(
        destination,
        dtype=np.float32,
    )

    # Ensure invalid numerical values are represented
    # consistently as NaN.
    destination[
        ~np.isfinite(destination)
    ] = np.nan

    validate_raster(
        destination,
        _metadata_from_reference(
            reference_metadata,
            shape=destination.shape,
            transform=reference_metadata["transform"],
        ),
        label="aligned raster",
    )

    return destination


# ============================================================
# ALIGN BAND
# ============================================================

def align_band_to_reference(
    band_array,
    band_metadata,
    reference_array,
    reference_metadata,
    resampling=Resampling.bilinear,
):
    """
    Align a raster band to an existing reference raster.

    If the band already has the same:
        - shape
        - CRS
        - transform

    it is returned without resampling.

    Otherwise it is reprojected/resampled to the exact
    reference grid.

    Returns:
        aligned numpy array

    Notes
    -----
    The public return type intentionally remains a numpy
    array to preserve compatibility with the existing
    application architecture.
    """

    validate_raster(
        band_array,
        band_metadata,
        label="band to align",
    )

    validate_raster(
        reference_array,
        reference_metadata,
        label="reference raster",
    )

    band_array = np.asarray(
        band_array,
        dtype=np.float32,
    )

    reference_array = np.asarray(
        reference_array,
        dtype=np.float32,
    )

    same_shape = (
        band_array.shape
        == reference_array.shape
    )

    same_transform = _transforms_close(
        band_metadata.get("transform"),
        reference_metadata.get("transform"),
    )

    same_crs = (
        band_metadata.get("crs")
        == reference_metadata.get("crs")
    )

    if (
        same_shape
        and same_transform
        and same_crs
    ):
        aligned = band_array.copy()

        aligned[
            ~np.isfinite(aligned)
        ] = np.nan

        validate_raster_pair(
            aligned,
            reference_array,
            _metadata_from_reference(
                reference_metadata,
                shape=aligned.shape,
                transform=reference_metadata["transform"],
            ),
            reference_metadata,
            label_a="aligned band",
            label_b="reference raster",
        )

        return aligned

    aligned = resample_to_reference(
        source_array=band_array,
        source_metadata=band_metadata,
        reference_metadata=reference_metadata,
        resampling=resampling,
    )

    aligned_metadata = _metadata_from_reference(
        reference_metadata,
        shape=aligned.shape,
        transform=reference_metadata["transform"],
    )

    validate_raster_pair(
        aligned,
        reference_array,
        aligned_metadata,
        reference_metadata,
        label_a="aligned band",
        label_b="reference raster",
    )

    return aligned


# ============================================================
# SAFE NORMALIZATION
# ============================================================

def normalize_image(
    image,
):
    """
    Normalize an image to the 0-1 range.

    Robust percentile normalization prevents a small number
    of extreme pixels from dominating visualization.

    Invalid pixels remain represented as zero because this
    function is intended for display imagery.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    valid = image[
        np.isfinite(image)
    ]

    if valid.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    low = np.percentile(
        valid,
        2,
    )

    high = np.percentile(
        valid,
        98,
    )

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    normalized = (
        image - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    normalized[
        ~np.isfinite(normalized)
    ] = 0.0

    return normalized.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# STACK BANDS
# ============================================================

def stack_bands(
    bands,
):
    """
    Stack multiple already-aligned bands.

    All bands must:
        - exist;
        - be 2D;
        - have identical shape.

    Returns:
        numpy array with shape (height, width, bands)
    """

    if not bands:
        raise ValueError(
            "No bands provided."
        )

    arrays = [
        np.asarray(
            band,
            dtype=np.float32,
        )
        for band in bands
    ]

    for index, band in enumerate(arrays):

        validate_raster(
            band,
            label=f"band[{index}]",
        )

    shapes = {
        band.shape
        for band in arrays
    }

    if len(shapes) != 1:
        raise RasterValidationError(
            "Cannot stack rasters with different "
            "spatial dimensions.\n\n"
            + "\n".join(
                f"Band {i}: {band.shape}"
                for i, band in enumerate(arrays)
            )
        )

    return np.stack(
        arrays,
        axis=-1,
    )