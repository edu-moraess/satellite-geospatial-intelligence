"""
Satellite Geospatial Intelligence
Configuration
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUT_DIR / "figures"

MAPS_DIR = OUTPUT_DIR / "maps"


# ============================================================
# PLANETARY COMPUTER
# ============================================================

PLANETARY_COMPUTER_STAC = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

SENTINEL_COLLECTION = "sentinel-2-l2a"


# ============================================================
# SENTINEL-2 BANDS
# ============================================================

AVAILABLE_BANDS = {
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B08": "NIR",
    "B11": "SWIR",
}