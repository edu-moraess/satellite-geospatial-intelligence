"""
Land Cover Classification
=========================

Rule-based multispectral land-cover classifier.

Classes:

0 -> Other
1 -> Vegetation
2 -> Water
3 -> Built-up
4 -> Bare Soil

This is a baseline classifier.

It intentionally does NOT claim to be
a trained machine-learning model.

Raster integrity is delegated to
src.raster_validation.
"""

from __future__ import annotations

import numpy as np

from src.raster_validation import (
    RasterValidationError,
    validate_raster,
)


# ============================================================
# CLASS IDS
# ============================================================

OTHER = 0

VEGETATION = 1

WATER = 2

BUILT_UP = 3

BARE_SOIL = 4


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    OTHER: "Other",
    VEGETATION: "Vegetation",
    WATER: "Water",
    BUILT_UP: "Built-up",
    BARE_SOIL: "Bare Soil",
}


# ============================================================
# INTERNAL VALIDATION
# ============================================================

def _validate_index(
    index,
    label: str,
):
    """
    Validate one spectral-index raster.

    Classification requires each input to be:
        - present;
        - non-empty;
        - 2D;
        - containing finite pixels.
    """

    return validate_raster(
        index,
        label=label,
    )


def _validate_index_stack(
    ndvi,
    ndwi,
    ndbi,
):
    """
    Validate the three spectral-index rasters before
    classification.

    In addition to validating each raster independently,
    this function guarantees that all three arrays have
    identical spatial dimensions and at least one pixel
    is simultaneously valid across all three indices.

    No resampling, cropping or broadcasting is performed.
    """

    _validate_index(
        ndvi,
        "NDVI",
    )

    _validate_index(
        ndwi,
        "NDWI",
    )

    _validate_index(
        ndbi,
        "NDBI",
    )

    ndvi = np.asarray(
        ndvi,
        dtype=np.float32,
    )

    ndwi = np.asarray(
        ndwi,
        dtype=np.float32,
    )

    ndbi = np.asarray(
        ndbi,
        dtype=np.float32,
    )

    if not (
        ndvi.shape
        == ndwi.shape
        == ndbi.shape
    ):
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

    return (
        ndvi,
        ndwi,
        ndbi,
        common_valid,
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_land_cover(
    ndvi,
    ndwi,
    ndbi,
    red=None,
    nir=None,
):
    """
    Classify pixels using spectral indices.

    Priority:

    1. Water
    2. Vegetation
    3. Built-up
    4. Bare soil
    5. Other

    Parameters:
        ndvi:
            Normalized Difference Vegetation Index.

        ndwi:
            Normalized Difference Water Index.

        ndbi:
            Normalized Difference Built-up Index.

        red:
            Optional red band reserved for future refinement.

        nir:
            Optional NIR band reserved for future refinement.

    Returns:
        uint8 classification map.

    Notes:
        Pixels where one or more indices are invalid remain
        classified as OTHER. They are never used in the
        spectral decision rules.

    Raises:
        RasterValidationError:
            If the input indices are missing, malformed,
            empty or spatially incompatible.
    """

    (
        ndvi,
        ndwi,
        ndbi,
        valid,
    ) = _validate_index_stack(
        ndvi,
        ndwi,
        ndbi,
    )

    classification = np.full(
        ndvi.shape,
        OTHER,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    water = (
        valid
        & (ndwi > 0.20)
        & (ndvi < 0.40)
    )

    classification[
        water
    ] = WATER

    # --------------------------------------------------------
    # VEGETATION
    # --------------------------------------------------------

    vegetation = (
        valid
        & ~water
        & (ndvi > 0.35)
        & (ndvi > ndbi)
    )

    classification[
        vegetation
    ] = VEGETATION

    # --------------------------------------------------------
    # BUILT-UP
    # --------------------------------------------------------

    built_up = (
        valid
        & ~water
        & ~vegetation
        & (ndbi > 0.05)
        & (ndbi > ndvi)
    )

    classification[
        built_up
    ] = BUILT_UP

    # --------------------------------------------------------
    # BARE SOIL
    # --------------------------------------------------------

    bare_soil = (
        valid
        & ~water
        & ~vegetation
        & ~built_up
        & (ndvi >= 0.10)
        & (ndvi <= 0.35)
        & (ndbi <= 0.05)
    )

    classification[
        bare_soil
    ] = BARE_SOIL

    return classification


# ============================================================
# CLASSIFICATION VALIDATION
# ============================================================

def validate_classification(
    classification,
):
    """
    Validate a generated classification map.

    Classification maps are categorical rasters, therefore
    the expected structure is a non-empty 2D array.

    Returns:
        Diagnostic dictionary.
    """

    return validate_raster(
        classification,
        label="land-cover classification",
    )


# ============================================================
# CLASS MASK
# ============================================================

def get_class_mask(
    classification,
    class_id,
):
    """
    Return a boolean mask for a specific class.
    """

    validate_classification(
        classification
    )

    if class_id not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class id: {class_id}. "
            f"Expected one of: {list(CLASS_NAMES)}."
        )

    classification = np.asarray(
        classification
    )

    return (
        classification
        == class_id
    )


# ============================================================
# CLASS PIXEL COUNTS
# ============================================================

def calculate_class_counts(
    classification,
):
    """
    Count pixels belonging to each class.
    """

    validate_classification(
        classification
    )

    classification = np.asarray(
        classification
    )

    total_valid = np.sum(
        classification != OTHER
    )

    counts = {}

    for class_id, name in (
        CLASS_NAMES.items()
    ):

        count = int(
            np.sum(
                classification
                == class_id
            )
        )

        counts[name] = count

    counts["_valid_pixels"] = int(
        total_valid
    )

    return counts


# ============================================================
# CLASS PERCENTAGES
# ============================================================

def calculate_class_percentages(
    classification,
):
    """
    Calculate percentage of each class.

    Percentages are calculated over all classified pixels,
    including Other.
    """

    validate_classification(
        classification
    )

    classification = np.asarray(
        classification
    )

    total_pixels = (
        classification.size
    )

    if total_pixels == 0:

        return {
            name: 0.0
            for name in CLASS_NAMES.values()
        }

    percentages = {}

    for class_id, name in (
        CLASS_NAMES.items()
    ):

        count = np.sum(
            classification
            == class_id
        )

        percentages[name] = (
            float(count)
            / float(total_pixels)
            * 100.0
        )

    return percentages