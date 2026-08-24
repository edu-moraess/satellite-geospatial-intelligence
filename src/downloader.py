"""
Satellite Geospatial Intelligence
----------------------------------

Satellite-2 band downloader.

Features:
- Downloads selected AOI only
- Creates scene directories safely
- Handles existing files/directories
- Reuses already downloaded bands
- Downloads B02, B03, B04, B08 and B11
"""

from pathlib import Path

import rasterio

from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


# ============================================================
# ENSURE OUTPUT DIRECTORY
# ============================================================

def ensure_output_directory(
    output_directory: Path,
):
    """
    Safely create the output directory.

    If a file exists where the directory should be,
    remove the conflicting file.
    """

    output_directory = Path(
        output_directory
    )

    # --------------------------------------------------------
    # EXISTING PATH
    # --------------------------------------------------------

    if output_directory.exists():

        # Path exists but is a file
        if output_directory.is_file():

            output_directory.unlink()

        # Path exists but is not a directory
        elif not output_directory.is_dir():

            raise RuntimeError(
                "Output path exists but is "
                "neither a file nor a directory: "
                f"{output_directory}"
            )

    # --------------------------------------------------------
    # CREATE DIRECTORY
    # --------------------------------------------------------

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


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

    output_directory = (
        ensure_output_directory(
            output_directory
        )
    )

    # --------------------------------------------------------
    # CHECK ASSET
    # --------------------------------------------------------

    asset = item.assets.get(
        band_name
    )

    if asset is None:

        raise ValueError(
            f"Band {band_name} is not available "
            f"in scene {item.id}."
        )

    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_path = (
        output_directory
        / f"{band_name}.tif"
    )

    # --------------------------------------------------------
    # REUSE EXISTING BAND
    # --------------------------------------------------------

    if (
        output_path.exists()
        and output_path.is_file()
    ):

        try:

            # Verify that the GeoTIFF can actually
            # be opened before reusing it.

            with rasterio.open(
                output_path
            ) as src:

                _ = src.width
                _ = src.height

            return output_path

        except Exception:

            # Corrupted/incomplete file.
            output_path.unlink(
                missing_ok=True
            )

    # --------------------------------------------------------
    # OPEN REMOTE SENTINEL RASTER
    # --------------------------------------------------------

    with rasterio.open(
        asset.href
    ) as src:

        # ----------------------------------------------------
        # TRANSFORM AOI TO RASTER CRS
        # ----------------------------------------------------

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
        # VALIDATE WINDOW
        # ----------------------------------------------------

        if (
            window.width <= 0
            or window.height <= 0
        ):

            raise ValueError(
                f"Selected area does not overlap "
                f"the raster for band {band_name}."
            )

        # ----------------------------------------------------
        # READ PIXELS
        # ----------------------------------------------------

        data = src.read(
            1,
            window=window,
        )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        transform = (
            src.window_transform(
                window
            )
        )

        # ----------------------------------------------------
        # PROFILE
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
    Download all bands required by the project.

    B02 -> Blue
    B03 -> Green
    B04 -> Red
    B08 -> NIR
    B11 -> SWIR
    """

    output_directory = (
        ensure_output_directory(
            output_directory
        )
    )

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