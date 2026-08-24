"""
Satellite Geospatial Intelligence
----------------------------------

Stage 1:
Automatic Sentinel-2 L2A acquisition.

The user defines:
- center latitude
- center longitude
- area size
- date range
- maximum cloud coverage

The script searches the Microsoft Planetary Computer
STAC catalog and downloads the requested Sentinel-2 bands.

Bands:
B02 -> Blue
B03 -> Green
B04 -> Red
B08 -> Near Infrared (NIR)
"""

from pathlib import Path
from datetime import datetime

import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER CONFIGURATION
# ============================================================

# Example: São Paulo
LATITUDE = -23.5505
LONGITUDE = -46.6333

# Area around the center point.
# 0.05 degrees is approximately a few kilometers.
AREA_SIZE = 0.05

# Date range
START_DATE = "2026-01-01"
END_DATE = "2026-08-23"

# Maximum acceptable cloud coverage
MAX_CLOUD_COVER = 10


# ============================================================
# SATELLITE CATALOG
# ============================================================

CATALOG_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)


# ============================================================
# CREATE BOUNDING BOX
# ============================================================

def create_bbox(
    latitude: float,
    longitude: float,
    area_size: float,
):
    """
    Create a geographic bounding box around
    the selected latitude and longitude.
    """

    half = area_size / 2

    min_lon = longitude - half
    min_lat = latitude - half
    max_lon = longitude + half
    max_lat = latitude + half

    return [
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ]


# ============================================================
# SEARCH SENTINEL-2
# ============================================================

def search_sentinel(
    bbox,
    start_date,
    end_date,
    max_cloud_cover,
):
    """
    Search Sentinel-2 Level-2A scenes.
    """

    catalog = pystac_client.Client.open(
        CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_cover
            }
        },
    )

    items = list(search.items())

    if not items:
        raise RuntimeError(
            "No Sentinel-2 images were found "
            "for the selected region/date/cloud criteria."
        )

    # Lowest cloud coverage first
    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100
        )
    )

    return items


# ============================================================
# PRINT SEARCH RESULTS
# ============================================================

def print_results(items):
    """
    Display available satellite scenes.
    """

    print("\n" + "=" * 70)
    print("SENTINEL-2 SCENES FOUND")
    print("=" * 70)

    for index, item in enumerate(items[:10], start=1):

        cloud_cover = item.properties.get(
            "eo:cloud_cover",
            "N/A"
        )

        date = item.datetime

        print(
            f"{index:02d} | "
            f"{date} | "
            f"Clouds: {cloud_cover:.2f}% | "
            f"{item.id}"
        )

    print("=" * 70)


# ============================================================
# DOWNLOAD / CROP BAND
# ============================================================

def download_band(
    item,
    band_name,
    output_path,
    bbox,
):
    """
    Download only the requested geographic window
    from a Sentinel-2 band.
    """

    asset = item.assets.get(band_name)

    if asset is None:
        raise RuntimeError(
            f"Band {band_name} not available "
            f"in satellite scene."
        )

    print(f"\nDownloading {band_name}...")

    with rasterio.open(asset.href) as src:

        # Convert geographic bbox into
        # the raster's coordinate system.
        raster_bbox = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox,
        )

        window = from_bounds(
            *raster_bbox,
            transform=src.transform,
        )

        window = window.round_offsets().round_lengths()

        data = src.read(
            1,
            window=window,
        )

        transform = src.window_transform(window)

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

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(data, 1)

    print(f"Saved: {output_path}")


# ============================================================
# DOWNLOAD ALL REQUIRED BANDS
# ============================================================

def download_required_bands(
    item,
    bbox,
):
    """
    Download B02, B03, B04 and B08.
    """

    output_directory = (
        RAW_DIR / item.id
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    bands = [
        "B02",
        "B03",
        "B04",
        "B08",
    ]

    for band in bands:

        output_path = (
            output_directory
            / f"{band}.tif"
        )

        download_band(
            item=item,
            band_name=band,
            output_path=output_path,
            bbox=bbox,
        )

    return output_directory


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SATELLITE GEOSPATIAL INTELLIGENCE")
    print("Sentinel-2 Automatic Acquisition")
    print("=" * 70)

    bbox = create_bbox(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        area_size=AREA_SIZE,
    )

    print("\nArea of interest:")
    print(f"Latitude : {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")
    print(f"BBox     : {bbox}")

    print("\nSearching Sentinel-2...")

    items = search_sentinel(
        bbox=bbox,
        start_date=START_DATE,
        end_date=END_DATE,
        max_cloud_cover=MAX_CLOUD_COVER,
    )

    print_results(items)

    # Best available scene
    selected_item = items[0]

    print("\nSelected scene:")
    print(selected_item.id)

    print(
        "Acquisition date:",
        selected_item.datetime
    )

    print(
        "Cloud coverage:",
        selected_item.properties.get(
            "eo:cloud_cover"
        ),
        "%"
    )

    output_directory = download_required_bands(
        item=selected_item,
        bbox=bbox,
    )

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Files saved in:\n{output_directory}"
    )

    print("\nBands:")
    print("B02 -> Blue")
    print("B03 -> Green")
    print("B04 -> Red")
    print("B08 -> NIR")


if __name__ == "__main__":
    main()