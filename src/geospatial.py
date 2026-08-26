"""
Geospatial utilities for Satellite Geospatial Intelligence.

Responsibilities:
- Read GeoTIFF bands safely.
- Preserve raster spatial metadata.
- Align bands to a reference raster grid.
- Resample bands when resolution/grid differs.
- Normalize imagery for visualization.
- Stack already-aligned bands.

Scientific processing is intentionally kept outside this module.
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
)


# ============================================================
# READ BAND
# ============================================================

def read_band(path):
    """
    Read a single-band GeoTIFF.

    Returns:
        tuple:
            array: np.ndarray with float32 values.
            metadata: dictionary containing spatial metadata.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        RasterValidationError:
            If the raster cannot be used safely.
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
            data = src.read(1).astype(
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
            }

    except RasterValidationError:
        raise

    except Exception as error:
        raise RasterValidationError(
            f"Unable to read raster: {path}\n\n"
            f"Rasterio error: {error}"
        ) from error

    validate_raster(
        data,
        metadata=metadata,
        label=path.name,
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

    This is required when Sentinel-2 bands have different
    native spatial resolutions.

    Typical example:

        B02 = 10 m
        B03 = 10 m
        B04 = 10 m
        B08 = 10 m
        B11 = 20 m

    B11 must therefore be resampled to the B04/B08 grid
    before NDBI or other pixel-wise calculations.

    Returns:
        np.ndarray with the reference height/width.
    """

    source_array = np.asarray(
        source_array,
        dtype=np.float32,
    )

    validate_raster(
        source_array,
        metadata=source_metadata,
        label="source raster",
    )

    if source_metadata is None:
        raise RasterValidationError(
            "Source raster metadata is required for "
            "geospatial resampling."
        )

    if reference_metadata is None:
        raise RasterValidationError(
            "Reference raster metadata is required for "
            "geospatial resampling."
        )

    source_transform = source_metadata.get(
        "transform"
    )

    source_crs = source_metadata.get(
        "crs"
    )

    reference_transform = reference_metadata.get(
        "transform"
    )

    reference_crs = reference_metadata.get(
        "crs"
    )

    reference_height = reference_metadata.get(
        "height"
    )

    reference_width = reference_metadata.get(
        "width"
    )

    if source_transform is None:
        raise RasterValidationError(
            "Source raster has no affine transform."
        )

    if source_crs is None:
        raise RasterValidationError(
            "Source raster has no CRS."
        )

    if reference_transform is None:
        raise RasterValidationError(
            "Reference raster has no affine transform."
        )

    if reference_crs is None:
        raise RasterValidationError(
            "Reference raster has no CRS."
        )

    if not reference_height or not reference_width:
        raise RasterValidationError(
            "Reference raster has invalid dimensions."
        )

    destination = np.full(
        (
            int(reference_height),
            int(reference_width),
        ),
        np.nan,
        dtype=np.float32,
    )

    source_nodata = source_metadata.get(
        "nodata"
    )

    try:
        reproject(
            source=source_array,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            src_nodata=source_nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    except Exception as error:
        raise RasterValidationError(
            "Raster resampling failed.\n\n"
            f"Source shape: {source_array.shape}\n"
            f"Target shape: "
            f"({reference_height}, {reference_width})\n"
            f"Error: {error}"
        ) from error

    validate_raster(
        destination,
        label="resampled raster",
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
):
    """
    Align a band to the exact spatial grid of a reference
    band.

    If shape, CRS and transform already match, the original
    array is returned unchanged.

    Otherwise the band is resampled to the reference grid.

    This function does not crop, broadcast or invent pixels.
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
        metadata=band_metadata,
        label="band to align",
    )

    validate_raster(
        reference_array,
        metadata=reference_metadata,
        label="reference band",
    )

    if band_metadata is None:
        raise RasterValidationError(
            "Band metadata is required for alignment."
        )

    if reference_metadata is None:
        raise RasterValidationError(
            "Reference metadata is required for alignment."
        )

    same_shape = (
        band_array.shape
        == reference_array.shape
    )

    same_transform = (
        band_metadata.get("transform")
        == reference_metadata.get("transform")
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
        return band_array

    return resample_to_reference(
        source_array=band_array,
        source_metadata=band_metadata,
        reference_metadata=reference_metadata,
        resampling=Resampling.bilinear,
    )


# ============================================================
# SAFE NORMALIZATION
# ============================================================

def normalize_image(
    image,
):
    """
    Normalize an image to the 0-1 range.

    Uses robust 2nd/98th percentile scaling so extreme
    values do not dominate visualization.

    NaN/Inf pixels remain NaN rather than being converted
    into fabricated measurements.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.size == 0:
        raise RasterValidationError(
            "Cannot normalize an empty image."
        )

    finite_mask = np.isfinite(
        image
    )

    valid = image[
        finite_mask
    ]

    if valid.size == 0:
        raise RasterValidationError(
            "Cannot normalize an image with no finite values."
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

    if high <= low:
        normalized = np.zeros_like(
            image,
            dtype=np.float32,
        )

        normalized[
            ~finite_mask
        ] = np.nan

        return normalized

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
        ~finite_mask
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

    Input:
        Iterable of 2D arrays.

    Returns:
        Array with shape:

            (height, width, bands)

    Raises:
        ValueError:
            If no bands are provided.

        RasterValidationError:
            If bands have incompatible shapes.
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

    for index, array in enumerate(
        arrays
    ):
        validate_raster(
            array,
            label=f"band {index}",
        )

    shapes = {
        array.shape
        for array in arrays
    }

    if len(shapes) != 1:
        raise RasterValidationError(
            "All bands must have the same "
            "spatial dimensions before stacking.\n\n"
            f"Received shapes: "
            f"{sorted(shapes)}"
        )

    return np.stack(
        arrays,
        axis=-1,
    )