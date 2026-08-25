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
from datetime import datetime, timezone

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
    Safely create the output directory for a scene.

    Handles the real-world failure mode that produced the
    original NotADirectoryError / FileExistsError: a previous
    crashed or interrupted download can leave a stray FILE
    sitting exactly where the scene directory
    (data/raw/<SCENE_ID>) needs to be created.

    Strategy:
        - If the exact target path exists as a file (not a
          directory), it is quarantined by renaming it with a
          timestamp suffix instead of being silently deleted,
          and the directory is created fresh. Nothing the user
          downloaded is ever destroyed, but the pipeline is
          never blocked by leftover junk either.
        - If a PARENT of the target path exists as a file
          (a more unusual, structural conflict that could
          affect other scenes too), we do NOT auto-heal - we
          raise a clear, specific RuntimeError so a human can
          decide what to do, instead of leaking a raw
          NotADirectoryError traceback.
    """

    output_directory = Path(
        output_directory
    )

    # ---- exact target path is a stray file: self-heal ----
    if output_directory.exists() and not output_directory.is_dir():

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        quarantined = output_directory.with_name(
            f"{output_directory.name}.corrupted.{timestamp}"
        )

        output_directory.rename(quarantined)

    # ---- walk up to find the first existing ancestor ----
    current = output_directory

    while not current.exists():

        current = current.parent

    if not current.is_dir():

        raise RuntimeError(
            "Filesystem conflict detected.\n\n"
            f"A parent path exists as a file:\n"
            f"{current}\n\n"
            "Expected a directory. This is a structural "
            "conflict that may affect other scenes, so it "
            "is not auto-healed. Please resolve it manually."
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
