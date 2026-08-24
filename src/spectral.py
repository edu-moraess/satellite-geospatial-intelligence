"""
Multispectral analysis.

Phase 2:
- NDVI
- NDWI
- NDBI
"""

from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# READ RASTER
# ============================================================

def read_raster(
    path,
):
    """
    Read raster data and metadata.
    """

    with rasterio.open(
        path
    ) as src:

        data = src.read(
            1
        ).astype(
            np.float32
        )

        profile = src.profile.copy()

    return data, profile


# ============================================================
# NORMALIZED DIFFERENCE
# ============================================================

def normalized_difference(
    band_a,
    band_b,
):
    """
    Generic normalized difference:

        (A - B) / (A + B)
    """

    band_a = band_a.astype(
        np.float32
    )

    band_b = band_b.astype(
        np.float32
    )

    denominator = (
        band_a + band_b
    )

    result = np.full(
        band_a.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(band_a)
        &
        np.isfinite(band_b)
        &
        (denominator != 0)
    )

    result[valid] = (
        band_a[valid]
        - band_b[valid]
    ) / denominator[valid]

    return result


# ============================================================
# NDVI
# ============================================================

def calculate_ndvi(
    red,
    nir,
):
    """
    NDVI:

        (NIR - RED)
        ------------
        (NIR + RED)

    Sentinel-2:
        RED = B04
        NIR = B08
    """

    return normalized_difference(
        nir,
        red,
    )


# ============================================================
# NDWI
# ============================================================

def calculate_ndwi(
    green,
    nir,
):
    """
    NDWI:

        (GREEN - NIR)
        --------------
        (GREEN + NIR)

    Sentinel-2:
        GREEN = B03
        NIR   = B08
    """

    return normalized_difference(
        green,
        nir,
    )


# ============================================================
# NDBI
# ============================================================

def calculate_ndbi(
    nir,
    swir,
):
    """
    NDBI:

        (SWIR - NIR)
        -------------
        (SWIR + NIR)

    Sentinel-2:
        NIR  = B08
        SWIR = B11
    """

    return normalized_difference(
        swir,
        nir,
    )


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    reference_path,
    output_path,
):
    """
    Save spectral index as GeoTIFF.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        reference_path
    ) as src:

        profile = src.profile.copy()

    profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": -9999,
            "compress": "deflate",
        }
    )

    output_data = np.where(
        np.isfinite(index),
        index,
        -9999,
    ).astype(
        np.float32
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            output_data,
            1,
        )

    return output_path