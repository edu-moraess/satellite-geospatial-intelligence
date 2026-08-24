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
"""

from __future__ import annotations

import numpy as np


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

    Returns:
        uint8 classification map
    """

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

        raise ValueError(
            "NDVI, NDWI and NDBI must "
            "have identical dimensions."
        )

    classification = np.full(
        ndvi.shape,
        OTHER,
        dtype=np.uint8,
    )

    valid = (
        np.isfinite(ndvi)
        & np.isfinite(ndwi)
        & np.isfinite(ndbi)
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
# CLASS MASK
# ============================================================

def get_class_mask(
    classification,
    class_id,
):
    """
    Return a boolean mask for a specific class.
    """

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

    Percentages are calculated over all
    classified pixels, including Other.
    """

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