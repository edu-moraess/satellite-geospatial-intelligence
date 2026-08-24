"""
Satellite Geospatial Intelligence
----------------------------------
Stage 1: Satellite image reader

Reads a GeoTIFF satellite image and displays
basic raster and geospatial metadata.

This module does NOT perform AI or spectral analysis yet.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Change this filename when you add your satellite image.
IMAGE_PATH = DATA_DIR / "satellite_image.tif"


# ============================================================
# SATELLITE READER
# ============================================================

def read_satellite_image(image_path: Path):
    """
    Read a GeoTIFF satellite image and return
    the raster data and metadata.
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Satellite image not found: {image_path}\n"
            "Place a .tif/.tiff file inside data/raw/"
        )

    with rasterio.open(image_path) as src:

        image = src.read()

        metadata = {
            "bands": src.count,
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": src.res,
            "dtype": src.dtypes,
        }

    return image, metadata


# ============================================================
# METADATA DISPLAY
# ============================================================

def print_metadata(metadata: dict):
    """Print basic satellite image information."""

    print("\n" + "=" * 60)
    print("SATELLITE IMAGE INFORMATION")
    print("=" * 60)

    print(f"Number of bands : {metadata['bands']}")
    print(f"Width           : {metadata['width']} pixels")
    print(f"Height          : {metadata['height']} pixels")
    print(f"CRS             : {metadata['crs']}")
    print(f"Resolution      : {metadata['resolution']}")
    print(f"Data type       : {metadata['dtype']}")

    print("\nGeographical bounds:")
    print(f"Left            : {metadata['bounds'].left}")
    print(f"Bottom          : {metadata['bounds'].bottom}")
    print(f"Right           : {metadata['bounds'].right}")
    print(f"Top             : {metadata['bounds'].top}")

    print("=" * 60)


# ============================================================
# SIMPLE VISUALIZATION
# ============================================================

def show_first_band(image):
    """
    Display the first raster band.

    This is intentionally simple.
    RGB and false-color composites will be added later.
    """

    first_band = image[0]

    plt.figure(figsize=(10, 8))

    plt.imshow(first_band, cmap="gray")

    plt.title("Satellite Image — First Band")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nSatellite Geospatial Intelligence Lab")
    print("Stage 1 — Satellite Image Reader")

    image, metadata = read_satellite_image(IMAGE_PATH)

    print_metadata(metadata)

    print("\nLoaded raster shape:")
    print(image.shape)

    show_first_band(image)


if __name__ == "__main__":
    main()