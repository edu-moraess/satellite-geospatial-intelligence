"""
Satellite Geospatial Intelligence
==================================

Robust Sentinel-2 downloader.

Features:
- Planetary Computer signed assets
- AOI window download
- Retry mechanism
- Safe directory creation
- GeoTIFF validation
- Existing-file reuse
- Remote raster error handling
"""

from pathlib import Path
import time

import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


# ============================================================
# CONSTANTS
# ============================================================

MAX_RETRIES = 3

RETRY_DELAY_SECONDS = 2


# ============================================================
# DIRECTORY
# ============================================================

def ensure_output_directory(
    output_directory: Path,
):
    """
    Safely create the output directory.
    """

    output_directory = Path(
        output_directory
    )

    current = output_directory

    while not current.exists():

        current = current.parent

    if not current.is_dir():

        raise RuntimeError(
            "Filesystem conflict detected.\n\n"
            f"Path exists as a file:\n"
            f"{current}\n\n"
            "Expected a directory."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


# ============================================================
# VALIDATE GEOTIFF
# ============================================================

def is_valid_geotiff(
    path: Path,
):
    """
    Check if an existing GeoTIFF is readable.
    """

    if not path.exists():
        return False

    if not path.is_file():
        return False

    try:

        with rasterio.open(
            path
        ) as src:

            return (
                src.width > 0
                and src.height > 0
            )

    except Exception:

        return False


# ============================================================
# SAFE REMOTE READ
# ============================================================

def read_remote_window(
    href,
    window,
):
    """
    Read a raster window from a remote asset.

    Uses retries because remote cloud-hosted
    rasters can occasionally fail during HTTP
    range requests.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_MAX_RETRY=3,
                GDAL_HTTP_RETRY_DELAY=1,
            ):

                with rasterio.open(
                    href
                ) as src:

                    data = src.read(
                        1,
                        window=window,
                    )

                    transform = (
                        src.window_transform(
                            window
                        )
                    )

                    profile = (
                        src.profile.copy()
                    )

                    profile.update(
                        {
                            "height": data.shape[0],
                            "width": data.shape[1],
                            "transform": transform,
                            "count": 1,
                            "compress": "deflate",
                        }
                    )

                    return (
                        data,
                        transform,
                        profile,
                    )

        except Exception as error:

            last_error = error

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY_SECONDS
                    * attempt
                )

    raise RuntimeError(
        "Unable to read the remote Sentinel-2 "
        "asset after multiple attempts.\n\n"
        f"Last error: {last_error}"
    )


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
    Download one Sentinel-2 band for the AOI.
    """

    output_directory = (
        ensure_output_directory(
            output_directory
        )
    )

    # --------------------------------------------------------
    # FIND ASSET
    # --------------------------------------------------------

    asset = item.assets.get(
        band_name
    )

    if asset is None:

        raise ValueError(
            f"Band {band_name} was not found "
            f"in scene {item.id}."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_path = (
        output_directory
        / f"{band_name}.tif"
    )

    # --------------------------------------------------------
    # REUSE
    # --------------------------------------------------------

    if is_valid_geotiff(
        output_path
    ):

        return output_path

    # --------------------------------------------------------
    # REMOVE INVALID FILE
    # --------------------------------------------------------

    if output_path.exists():

        output_path.unlink(
            missing_ok=True
        )

    # --------------------------------------------------------
    # REMOTE ASSET
    # --------------------------------------------------------

    href = asset.href

    if not href:

        raise ValueError(
            f"Asset URL unavailable for "
            f"{band_name}."
        )

    # --------------------------------------------------------
    # OPEN REMOTE DATASET
    # --------------------------------------------------------

    try:

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            GDAL_HTTP_MAX_RETRY=3,
            GDAL_HTTP_RETRY_DELAY=1,
        ):

            with rasterio.open(
                href
            ) as src:

                # --------------------------------------------
                # TRANSFORM AOI
                # --------------------------------------------

                raster_bbox = (
                    transform_bounds(
                        "EPSG:4326",
                        src.crs,
                        *bbox,
                    )
                )

                # --------------------------------------------
                # CLIP TO RASTER BOUNDS
                # --------------------------------------------

                left = max(
                    raster_bbox[0],
                    src.bounds.left,
                )

                bottom = max(
                    raster_bbox[1],
                    src.bounds.bottom,
                )

                right = min(
                    raster_bbox[2],
                    src.bounds.right,
                )

                top = min(
                    raster_bbox[3],
                    src.bounds.top,
                )

                if (
                    left >= right
                    or bottom >= top
                ):

                    raise ValueError(
                        f"AOI does not overlap "
                        f"band {band_name}."
                    )

                # --------------------------------------------
                # WINDOW
                # --------------------------------------------

                window = from_bounds(
                    left,
                    bottom,
                    right,
                    top,
                    transform=src.transform,
                )

                window = (
                    window
                    .round_offsets()
                    .round_lengths()
                )

                # --------------------------------------------
                # VALIDATE
                # --------------------------------------------

                if (
                    window.width <= 0
                    or window.height <= 0
                ):

                    raise ValueError(
                        f"Invalid raster window "
                        f"for {band_name}."
                    )

                # --------------------------------------------
                # READ WITH RETRIES
                # --------------------------------------------

                data = None

                last_error = None

                for attempt in range(
                    1,
                    MAX_RETRIES + 1,
                ):

                    try:

                        data = src.read(
                            1,
                            window=window,
                        )

                        break

                    except Exception as error:

                        last_error = error

                        if attempt < MAX_RETRIES:

                            time.sleep(
                                RETRY_DELAY_SECONDS
                                * attempt
                            )

                if data is None:

                    raise RuntimeError(
                        f"Failed to read "
                        f"{band_name} after "
                        f"{MAX_RETRIES} attempts.\n\n"
                        f"Last error: {last_error}"
                    )

                # --------------------------------------------
                # TRANSFORM
                # --------------------------------------------

                transform = (
                    src.window_transform(
                        window
                    )
                )

                # --------------------------------------------
                # PROFILE
                # --------------------------------------------

                profile = (
                    src.profile.copy()
                )

                profile.update(
                    {
                        "driver": "GTiff",
                        "height": data.shape[0],
                        "width": data.shape[1],
                        "transform": transform,
                        "count": 1,
                        "compress": "deflate",
                        "dtype": str(
                            data.dtype
                        ),
                    }
                )

    except Exception as error:

        raise RuntimeError(
            f"Failed downloading "
            f"{band_name}.\n\n"
            f"Scene: {item.id}\n"
            f"Asset: {href}\n\n"
            f"Error: {error}"
        ) from error

    # --------------------------------------------------------
    # WRITE LOCAL FILE
    # --------------------------------------------------------

    try:

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                data,
                1,
            )

    except Exception as error:

        output_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Failed writing "
            f"{output_path}.\n\n"
            f"Error: {error}"
        ) from error

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if not is_valid_geotiff(
        output_path
    ):

        output_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Downloaded file is invalid: "
            f"{output_path}"
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
    """

    output_directory = (
        ensure_output_directory(
            output_directory
        )
    )

    required_bands = [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
    ]

    downloaded = {}

    for band in required_bands:

        downloaded[band] = (
            download_band(
                item=item,
                band_name=band,
                bbox=bbox,
                output_directory=(
                    output_directory
                ),
            )
        )

    return downloaded