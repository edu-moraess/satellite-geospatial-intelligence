"""
Geospatial utilities for Satellite Geospatial Intelligence.
"""

from pathlib import Path

import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject


# ============================================================
# READ BAND
# ============================================================

def read_band(path):
    """
    Read a GeoTIFF band.

    Returns:
        array, metadata
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Band file not found: {path}"
        )

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

    return data, metadata


# ============================================================
# NORMALIZE BAND TO REFERENCE GRID
# ============================================================

def resample_to_reference(
    source_array,
    source_metadata,
    reference_metadata,
    resampling=Resampling.bilinear,
):
    """
    Resample a spectral band to the spatial grid
    of another reference band.

    This is required because Sentinel-2 bands can
    have different spatial resolutions.

    Example:

        B08 = 10 m
        B11 = 20 m

    B11 must be resampled to B08 before NDBI.
    """

    destination = np.full(
        (
            reference_metadata["height"],
            reference_metadata["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    source_nodata = (
        source_metadata.get("nodata")
    )

    destination_nodata = np.nan

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
        dst_nodata=destination_nodata,
        resampling=resampling,
    )

    return destination


# ============================================================
# ALIGN BANDS
# ============================================================

def align_band_to_reference(
    band_array,
    band_metadata,
    reference_array,
    reference_metadata,
):
    """
    Make sure a band has exactly the same shape
    and spatial grid as the reference band.
    """

    same_shape = (
        band_array.shape
        == reference_array.shape
    )

    same_transform = (
        band_metadata["transform"]
        == reference_metadata["transform"]
    )

    same_crs = (
        band_metadata["crs"]
        == reference_metadata["crs"]
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

    Robust percentile normalization prevents
    a few extreme pixels from destroying the
    visualization.
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

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ============================================================
# STACK BANDS
# ============================================================

def stack_bands(
    bands,
):
    """
    Stack multiple bands into a single
    numpy array.

    All arrays must already be aligned.
    """

    if not bands:

        raise ValueError(
            "No bands provided."
        )

    shapes = {
        np.asarray(
            band
        ).shape
        for band in bands
    }

    if len(shapes) != 1:

        raise ValueError(
            "All bands must have the same "
            "spatial dimensions before stacking."
        )

    return np.stack(
        bands,
        axis=-1,
    )