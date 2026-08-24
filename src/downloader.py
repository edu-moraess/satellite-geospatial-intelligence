"""
Sentinel-2 band downloader.

Downloads only the geographic window
selected by the user.
"""

from pathlib import Path

import rasterio

from rasterio.windows import from_bounds

from rasterio.warp import transform_bounds


# ============================================================
# DOWNLOAD SINGLE BAND
# ============================================================

def download_band(
    item,
    band_name: str,
    bbox,
    output_directory: Path,
):
    """
    Download one Sentinel-2 band.
    """

    asset = item.assets.get(
        band_name
    )

    if asset is None:

        raise ValueError(
            f"Band {band_name} is not "
            "available in this scene."
        )

    # Create directory only when
    # the download actually happens.
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{band_name}.tif"
    )

    # --------------------------------------------------------
    # OPEN REMOTE RASTER
    # --------------------------------------------------------

    with rasterio.open(
        asset.href
    ) as src:

        # Transform AOI from WGS84
        # to the raster CRS.

        raster_bbox = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox,
        )

        # ----------------------------------------------------
        # CREATE RASTER WINDOW
        # ----------------------------------------------------

        window = from_bounds(
            *raster_bbox,
            transform=src.transform,
        )

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        # ----------------------------------------------------
        # READ DATA
        # ----------------------------------------------------

        data = src.read(
            1,
            window=window,
        )

        transform = (
            src.window_transform(
                window
            )
        )

        # ----------------------------------------------------
        # OUTPUT PROFILE
        # ----------------------------------------------------

        profile = src.profile.copy()

        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "count": 1,
                "compress": "deflate",
            }
        )

        # ----------------------------------------------------
        # SAVE GEOTIFF
        # ----------------------------------------------------

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                data,
                1,
            )

    return output_path


# ============================================================
# DOWNLOAD REQUIRED BANDS
# ============================================================

def download_required_bands(
    item,
    bbox,
    output_directory: Path,
):
    """
    Download all bands required by
    Phase 1 and Phase 2.

    B02 -> Blue
    B03 -> Green
    B04 -> Red
    B08 -> NIR
    B11 -> SWIR
    """

    bands = [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
    ]

    downloaded = {}

    for band in bands:

        downloaded[band] = download_band(
            item=item,
            band_name=band,
            bbox=bbox,
            output_directory=(
                output_directory
            ),
        )

    return downloaded