"""
Basic raster/geospatial utilities.
"""

import numpy as np
import rasterio


# ============================================================
# READ BAND
# ============================================================

def read_band(
    path,
):
    """
    Read a single-band GeoTIFF.
    """

    with rasterio.open(
        path
    ) as src:

        data = src.read(
            1
        )

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "resolution": src.res,
            "nodata": src.nodata,
        }

    return data, metadata


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(
    image,
):
    """
    Normalize image using percentile stretching.

    This is for visualization only.
    It does not modify the original satellite data.
    """

    image = image.astype(
        np.float32
    )

    valid = np.isfinite(
        image
    )

    if not np.any(valid):

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    min_value = np.nanpercentile(
        image[valid],
        2,
    )

    max_value = np.nanpercentile(
        image[valid],
        98,
    )

    if max_value <= min_value:

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    normalized = (
        image - min_value
    ) / (
        max_value - min_value
    )

    return np.clip(
        normalized,
        0,
        1,
    )