"""
Sentinel-2 band downloader.

Downloads only the selected geographic window
instead of unnecessarily downloading the entire scene.
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
    Download one Sentinel-2 band for the selected AOI.
    """

    asset = item.assets.get(
        band_name
    )

    if asset is None:

        raise ValueError(
            f"Band {band_name} is not available "
            f"in this Sentinel-2 scene."
        )

    # Create directory only when needed
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

        # Transform geographic coordinates
        # into the raster CRS.

        raster_bbox = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox,
        )

        # ----------------------------------------------------
        # CREATE WINDOW
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
        # READ PIXELS
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
        # CREATE OUTPUT PROFILE
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
        # WRITE GEOTIFF
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
    Download the four bands needed for Phase 1.

    B02 -> Blue
    B03 -> Green
    B04 -> Red
    B08 -> NIR
    """

    bands = [
        "B02",
        "B03",
        "B04",
        "B08",
    ]

    downloaded = {}

    for band in bands:

        downloaded[band] = download_band(
            item=item,
            band_name=band,
            bbox=bbox,
            output_directory=output_directory,
        )

    return downloaded