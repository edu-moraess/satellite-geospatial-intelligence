"""
Basic raster utilities.
"""

import numpy as np
import rasterio


# ============================================================
# READ RASTER
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
        ).astype(
            np.float32
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
# NORMALIZE FOR DISPLAY
# ============================================================

def normalize_image(
    image,
):
    """
    Percentile normalization.

    Used only for visualization.
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

    low = np.nanpercentile(
        image[valid],
        2,
    )

    high = np.nanpercentile(
        image[valid],
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
        0,
        1,
    )