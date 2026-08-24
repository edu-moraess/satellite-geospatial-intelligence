"""
Satellite Geospatial Intelligence
----------------------------------

Sentinel-2 downloader.

Responsibilities:

1. Validate the data directory.
2. Create scene directories.
3. Download selected Sentinel-2 bands.
4. Extract only the user's AOI.
5. Reuse valid files when possible.

Bands:

B02 -> Blue
B03 -> Green
B04 -> Red
B08 -> NIR
B11 -> SWIR
"""

from pathlib import Path

import rasterio

from rasterio.windows import from_bounds

from rasterio.warp import transform_bounds


# ============================================================
# ENSURE DIRECTORY
# ============================================================

def ensure_output_directory(
    output_directory: Path,
):
    """
    Safely create an output directory.

    Handles situations such as:

        data/raw
        data/raw/scene_id

    If a path component is accidentally a file,
    a clear error is generated instead of producing
    a confusing NotADirectoryError.
    """

    output_directory = Path(
        output_directory
    )

    # --------------------------------------------------------
    # CHECK EVERY PARENT
    # --------------------------------------------------------

    current = output_directory

    missing_parts = []

    while not current.exists():

        missing_parts.append(
            current
        )

        current = current.parent

    # --------------------------------------------------------
    # FIND EXISTING ANCESTOR
    # --------------------------------------------------------

    if current.exists() and not current.is_dir():

        raise RuntimeError(
            "❌ Filesystem conflict detected.\n\n"
            f"The path:\n"
            f"{current}\n\n"
            "exists as a FILE, but the application "
            "needs it to be a DIRECTORY.\n\n"
            "Please remove that file from GitHub "
            "and create the folder structure:\n\n"
            "data/raw/\n"
            "data/processed/\n"
        )

    # --------------------------------------------------------
    # CREATE MISSING DIRECTORIES
    # --------------------------------------------------------

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


# ============================================================
# VALIDATE EXISTING GEOTIFF
# ============================================================

def is_valid_geotiff(
    path: Path,
):
    """
    Check whether an existing file is a readable GeoTIFF.
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
# DOWNLOAD SINGLE BAND
# ============================================================

def download_band(
    item,
    band_name: str,
    bbox,
    output_directory: Path,
):
    """
    Download a single Sentinel-2 band.

    Only the selected AOI is downloaded.
    """

    # --------------------------------------------------------
    # ENSURE DIRECTORY
    # --------------------------------------------------------

    output_directory = (
        ensure_output_directory(
            output_directory
        )
    )

    # --------------------------------------------------------
    # FIND SENTINEL ASSET
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
    # REUSE EXISTING FILE
    # --------------------------------------------------------

    if is_valid_geotiff(
        output_path
    ):

        return output_path

    # --------------------------------------------------------
    # DELETE CORRUPTED FILE
    # --------------------------------------------------------

    if output_path.exists():

        output_path.unlink()

    # --------------------------------------------------------
    # OPEN REMOTE RASTER
    # --------------------------------------------------------

    with rasterio.open(
        asset.href
    ) as src:

        # ----------------------------------------------------
        # TRANSFORM BBOX
        # ----------------------------------------------------

        raster_bbox = (
            transform_bounds(
                "EPSG:4326",
                src.crs,
                *bbox,
            )
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
                f"The selected area does not "
                f"overlap band {band_name}."
            )

        # ----------------------------------------------------
        # READ PIXELS
        # ----------------------------------------------------

        data = src.read(
            1,
            window=window,
        )

        # ----------------------------------------------------
        # CREATE TRANSFORM
        # ----------------------------------------------------

        transform = (
            src.window_transform(
                window
            )
        )

        # ----------------------------------------------------
        # CREATE PROFILE
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
# DOWNLOAD ALL REQUIRED BANDS
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

    # --------------------------------------------------------
    # ENSURE SCENE DIRECTORY
    # --------------------------------------------------------

    output_directory = (
        ensure_output_directory(
            output_directory
        )
    )

    # --------------------------------------------------------
    # REQUIRED BANDS
    # --------------------------------------------------------

    bands = [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
    ]

    downloaded = {}

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    for band in bands:

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