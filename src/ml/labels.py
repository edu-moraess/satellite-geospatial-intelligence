"""ESA WorldCover label mapping for experimental ML land-cover classification."""
from __future__ import annotations

from typing import Mapping

import numpy as np

OTHER = 0
VEGETATION = 1
WATER = 2
BUILT_UP = 3
BARE_SOIL = 4

CLASS_NAMES = (
    "Other",
    "Vegetation",
    "Water",
    "Built-up",
    "Bare Soil",
)

WORLDCOVER_TO_CLASS: Mapping[int, int] = {
    10: VEGETATION,
    20: VEGETATION,
    30: VEGETATION,
    40: VEGETATION,
    50: BUILT_UP,
    60: BARE_SOIL,
    70: OTHER,
    80: WATER,
    90: VEGETATION,
    95: VEGETATION,
    100: VEGETATION,
}

WORLDCOVER_CODES = tuple(WORLDCOVER_TO_CLASS)


def map_worldcover_labels(labels: np.ndarray) -> np.ndarray:
    """Map ESA WorldCover codes to GEOCORE's five ML classes.

    Unknown codes are conservatively mapped to Other. The output dtype is
    uint8 and contains only the documented GEOCORE class IDs.
    """
    values = np.asarray(labels)
    result = np.full(values.shape, OTHER, dtype=np.uint8)
    for source_code, target_class in WORLDCOVER_TO_CLASS.items():
        result[values == source_code] = target_class
    return result
