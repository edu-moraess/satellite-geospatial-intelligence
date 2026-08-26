"src/geospatial.py"

"""
Geospatial utilities for Satellite Geospatial Intelligence.

Responsibilities:
    - reading GeoTIFF bands;
    - raster reprojection;
    - raster alignment;
    - spatial-grid comparison;
    - safe image normalization;
    - aligned band stacking.

Scientific operations themselves do not live here.

This module is responsible for making sure arrays and their
spatial metadata are correctly represented before they reach
the analytical layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from .raster_validation import (
    RasterValidationError,
    transforms_close,
    validate_raster,
    validate_raster_pair,
)


# ============================================================
# READ BAND
# ============================================================

def read_band(path):
    """
    Read a single-band GeoTIFF.

    Returns:
        array, metadata
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

            data = src.read(
                1
            ).astype(
                np.float32
            )

            metadata = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "bounds": src.bounds,
                "profile": src.profile.copy(),
                "dtype": str(src.dtypes[0]),
                "driver": src.driver,
            }

    except RasterValidationError:
        raise

    except Exception as error:
        raise RasterValidationError(
            f"Failed to read raster: {path}\n\n"
            f"Error: {error}"
        ) from error

    validate_raster(
        data,
        metadata,
        label=path.name,
    )

    return data, metadata


# ============================================================
# GRID COMPARISON
# ============================================================

def same_spatial_grid(
    array_a,
    metadata_a,
    array_b,
    metadata_b,
    *,
    transform_tolerance: float = 1e-6,
) -> bool:
    """
    Return True when two rasters already share the same
    mathematical pixel grid.

    Checks:
        - shape;
        - CRS;
        - transform.

    This function does not modify either raster.
    """

    array_a = np.asarray(array_a)
    array_b = np.asarray(array_b)

    if array_a.shape != array_b.shape:
        return False

    crs_a = (
        metadata_a.get("crs")
        if metadata_a
        else None
    )

    crs_b = (
        metadata_b.get("crs")
        if metadata_b
        else None
    )

    if (
        crs_a is not None
        and crs_b is not None
        and crs_a != crs_b
    ):
        return False

    transform_a = (
        metadata_a.get("transform")
        if metadata_a
        else None
    )

    transform_b = (
        metadata_b.get("transform")
        if metadata_b
        else None
    )

    return transforms_close(
        transform_a,
        transform_b,
        tolerance=transform_tolerance,
    )


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
    Reproject/resample a raster onto the exact spatial grid
    represented by reference_metadata.

    The returned array has:
        - reference height;
        - reference width;
        - reference CRS;
        - reference transform.

    This function does not return metadata because the
    resulting grid is exactly the reference grid.
    """

    source_array = np.asarray(
        source_array,
        dtype=np.float32,
    )

    validate_raster(
        source_array,
        source_metadata,
        label="source raster",
    )

    required_source_keys = (
        "transform",
        "crs",
    )

    required_reference_keys = (
        "transform",
        "crs",
        "height",
        "width",
    )

    for key in required_source_keys:
        if source_metadata.get(key) is None:
            raise RasterValidationError(
                f"Source raster metadata is missing '{key}'."
            )

    for key in required_reference_keys:
        if reference_metadata.get(key) is None:
            raise RasterValidationError(
                f"Reference raster metadata is missing '{key}'."
            )

    height = int(
        reference_metadata["height"]
    )

    width = int(
        reference_metadata["width"]
    )

    if height <= 0 or width <= 0:
        raise RasterValidationError(
            "Reference raster has invalid dimensions: "
            f"{height} x {width}."
        )

    destination = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    source_nodata = source_metadata.get(
        "nodata"
    )

    # Rasterio's reprojection machinery can handle NaN
    # destinations. For source data without an explicit
    # nodata value, finite values are copied normally.
    reproject(
        source=source_array,
        destination=destination,
        src_transform=source_metadata[
            "transform"
        ],
        src_crs=source_metadata[
            "crs"
        ],
        dst_transform=reference_metadata[
            "transform"
        ],
        dst_crs=reference_metadata[
            "crs"
        ],
        src_nodata=source_nodata,
        dst_nodata=np.nan,
        resampling=resampling,
    )

    validate_raster(
        destination,
        label="aligned raster",
    )

    return destination


# ============================================================
# ALIGN BAND TO REFERENCE
# ============================================================

def align_band_to_reference(
    band_array,
    band_metadata,
    reference_array,
    reference_metadata,
    *,
    resampling=Resampling.bilinear,
):
    """
    Align one raster band to the exact grid of a reference
    raster.

    If both rasters are already aligned, the original array
    is returned without resampling.

    If not, the band is reprojected/resampled to the reference
    grid.

    The function preserves the historical API used by app.py:
        align_band_to_reference(...) -> ndarray
    """

    band_array = np.asarray(
        band_array,
        dtype=np.float32,
    )

    reference_array = np.asarray(
        reference_array,
        dtype=np.float32,
    )

    validate_raster(
        band_array,
        band_metadata,
        label="band",
    )

    validate_raster(
        reference_array,
        reference_metadata,
        label="reference band",
    )

    if same_spatial_grid(
        band_array,
        band_metadata,
        reference_array,
        reference_metadata,
    ):
        return band_array

    return resample_to_reference(
        source_array=band_array,
        source_metadata=band_metadata,
        reference_metadata=reference_metadata,
        resampling=resampling,
    )


# ============================================================
# ALIGN ARRAY + RETURN METADATA
# ============================================================

def align_array_with_metadata(
    source_array,
    source_metadata,
    reference_array,
    reference_metadata,
    *,
    resampling=Resampling.bilinear,
):
    """
    Align a raster to a reference grid and return both the
    resulting array and synchronized metadata.

    Returns:
        aligned_array, aligned_metadata

    This is especially useful for Change Detection, where
    the After scene must be represented on the exact grid
    of the Before scene.
    """

    source_array = np.asarray(
        source_array,
        dtype=np.float32,
    )

    reference_array = np.asarray(
        reference_array,
        dtype=np.float32,
    )

    validate_raster(
        source_array,
        source_metadata,
        label="source raster",
    )

    validate_raster(
        reference_array,
        reference_metadata,
        label="reference raster",
    )

    if same_spatial_grid(
        source_array,
        source_metadata,
        reference_array,
        reference_metadata,
    ):
        aligned = source_array
    else:
        aligned = resample_to_reference(
            source_array=source_array,
            source_metadata=source_metadata,
            reference_metadata=reference_metadata,
            resampling=resampling,
        )

    aligned_metadata = dict(
        reference_metadata
    )

    aligned_metadata.update(
        {
            "height": int(
                reference_array.shape[0]
            ),
            "width": int(
                reference_array.shape[1]
            ),
            "transform": reference_metadata[
                "transform"
            ],
            "crs": reference_metadata[
                "crs"
            ],
            "nodata": np.nan,
            "dtype": str(
                aligned.dtype
            ),
        }
    )

    validate_raster_pair(
        aligned,
        reference_array,
        aligned_metadata,
        reference_metadata,
        label_a="aligned raster",
        label_b="reference raster",
        require_same_dtype=False,
    )

    return aligned, aligned_metadata


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(
    image,
):
    """
    Normalize an image to [0, 1].

    Robust percentile normalization prevents a small number
    of extreme pixels from dominating visualization.
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

    low = float(
        np.percentile(
            valid,
            2,
        )
    )

    high = float(
        np.percentile(
            valid,
            98,
        )
    )

    if not np.isfinite(low) or not np.isfinite(high):
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    if high <= low:
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

    # Preserve invalid pixels as NaN instead of converting
    # them into artificial valid observations.
    normalized[
        ~np.isfinite(image)
    ] = np.nan

    return normalized.astype(
        np.float32
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
        - be non-empty;
        - be 2D;
        - share the same spatial dimensions.

    No automatic resampling is performed here.
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

    for index, array in enumerate(arrays):
        validate_raster(
            array,
            label=f"band[{index}]",
        )

    shapes = {
        array.shape
        for array in arrays
    }

    if len(shapes) != 1:
        raise RasterValidationError(
            "All bands must have the same spatial dimensions "
            "before stacking.\n\n"
            f"Detected shapes: {sorted(shapes)}"
        )

    return np.stack(
        arrays,
        axis=-1,
    )