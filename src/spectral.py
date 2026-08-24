"""
Satellite Geospatial Intelligence
----------------------------------

Phase 2:
Multispectral analysis.

Indices:
- NDVI -> Vegetation
- NDWI -> Water
- NDBI -> Built-up areas
"""

from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# READ RASTER
# ============================================================

def read_raster(path):
    """
    Read a single-band GeoTIFF.
    """

    with rasterio.open(path) as src:

        data = src.read(1).astype(
            np.float32
        )

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        nodata = src.nodata

    return (
        data,
        profile,
        transform,
        crs,
        nodata,
    )


# ============================================================
# SAFE NORMALIZED DIFFERENCE
# ============================================================

def normalized_difference(
    band_a,
    band_b,
):
    """
    Calculate:

        (A - B) / (A + B)

    Avoiding division by zero.
    """

    denominator = (
        band_a + band_b
    )

    result = np.full_like(
        band_a,
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
        -------------
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

    IMPORTANT:
    Sentinel-2 SWIR is B11.

    This requires downloading B11.
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
    Save calculated index as GeoTIFF.
    """

    with rasterio.open(
        reference_path
    ) as src:

        profile = src.profile.copy()

    profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": np.nan,
            "compress": "deflate",
        }
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            index.astype(
                np.float32
            ),
            1,
        )

    return output_path