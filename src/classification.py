from __future__ import annotations

"""
Land Cover Classification

Rule-based multispectral land-cover classifier.

Classes:
0 -> Other
1 -> Vegetation
2 -> Water
3 -> Built-up
4 -> Bare Soil

This is a baseline rule-based classifier.
It does NOT claim to be a trained machine-learning model.

Raster integrity is delegated to src.raster_validation.
"""

import numpy as np

from src.raster_validation import (
    RasterValidationError,
    validate_raster,
)

OTHER = 0
VEGETATION = 1
WATER = 2
BUILT_UP = 3
BARE_SOIL = 4

CLASS_NAMES = {
    OTHER: "Other",
    VEGETATION: "Vegetation",
    WATER: "Water",
    BUILT_UP: "Built-up",
    BARE_SOIL: "Bare Soil",
}


def _validate_index(index, label: str):
    """Validate a single spectral index."""
    return validate_raster(index, label=label)


def _validate_index_stack(ndvi, ndwi, ndbi):
    """
    Validate NDVI, NDWI and NDBI before classification.

    No resampling, cropping or broadcasting is performed here.
    """
    _validate_index(ndvi, "NDVI")
    _validate_index(ndwi, "NDWI")
    _validate_index(ndbi, "NDBI")

    ndvi = np.asarray(ndvi, dtype=np.float32)
    ndwi = np.asarray(ndwi, dtype=np.float32)
    ndbi = np.asarray(ndbi, dtype=np.float32)

    if not (ndvi.shape == ndwi.shape == ndbi.shape):
        raise RasterValidationError(
            "Spectral index shape mismatch.\n\n"
            f"NDVI: {ndvi.shape}\n"
            f"NDWI: {ndwi.shape}\n"
            f"NDBI: {ndbi.shape}\n\n"
            "All spectral indices must share the same "
            "spatial grid before land-cover classification."
        )

    common_valid = (
        np.isfinite(ndvi)
        & np.isfinite(ndwi)
        & np.isfinite(ndbi)
    )

    if not np.any(common_valid):
        raise RasterValidationError(
            "NDVI, NDWI and NDBI have no common valid pixels.\n\n"
            "The land-cover classifier cannot produce a valid "
            "classification from these inputs."
        )

    return ndvi, ndwi, ndbi, common_valid


def classify_land_cover(
    ndvi,
    ndwi,
    ndbi,
    red=None,
    nir=None,
):
    """
    Classify pixels using NDVI, NDWI and NDBI.

    Decision priority:

        1. Water
        2. Vegetation
        3. Built-up
        4. Bare Soil
        5. Other

    red and nir are accepted for forward compatibility.
    They are not currently used by the baseline classifier.
    """
    ndvi, ndwi, ndbi, valid = _validate_index_stack(ndvi, ndwi, ndbi)

    classification = np.full(ndvi.shape, OTHER, dtype=np.uint8)

    water = valid & (ndwi > 0.20) & (ndvi < 0.40)
    classification[water] = WATER

    vegetation = valid & ~water & (ndvi > 0.35) & (ndvi > ndbi)
    classification[vegetation] = VEGETATION

    built_up = valid & ~water & ~vegetation & (ndbi > 0.05) & (ndbi > ndvi)
    classification[built_up] = BUILT_UP

    bare_soil = (
        valid
        & ~water
        & ~vegetation
        & ~built_up
        & (ndvi >= 0.10)
        & (ndvi <= 0.35)
        & (ndbi <= 0.05)
    )
    classification[bare_soil] = BARE_SOIL

    return classification


def validate_classification(classification):
    """Validate a generated classification map."""
    return validate_raster(classification, label="land-cover classification")


def get_class_mask(classification, class_id):
    """Return a boolean mask for a specific class."""
    validate_classification(classification)

    if class_id not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class id: {class_id}. "
            f"Expected one of: {list(CLASS_NAMES)}."
        )

    classification = np.asarray(classification)
    return classification == class_id


def calculate_class_counts(classification):
    """Count pixels belonging to every class."""
    validate_classification(classification)

    classification = np.asarray(classification)
    counts = {}

    for class_id, name in CLASS_NAMES.items():
        count = int(np.sum(classification == class_id))
        counts[name] = count

    counts["_valid_pixels"] = int(np.sum(classification != OTHER))
    counts["_total_pixels"] = int(classification.size)

    return counts


def calculate_class_percentages(classification):
    """
    Calculate the percentage occupied by each class.

    Percentages are calculated over the complete raster,
    including pixels classified as Other.
    """
    validate_classification(classification)

    classification = np.asarray(classification)
    total_pixels = int(classification.size)

    if total_pixels == 0:
        return {name: 0.0 for name in CLASS_NAMES.values()}

    percentages = {}

    for class_id, name in CLASS_NAMES.items():
        count = int(np.sum(classification == class_id))
        percentages[name] = (float(count) / float(total_pixels)) * 100.0

    return percentages