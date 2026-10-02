import unittest

import numpy as np

from src.ml.labels import (
    BARE_SOIL,
    BUILT_UP,
    CLASS_NAMES,
    OTHER,
    VEGETATION,
    WATER,
    WORLDCOVER_CODES,
    WORLDCOVER_TO_CLASS,
    map_worldcover_labels,
)

class TestWorldCoverLabels(unittest.TestCase):
    def test_all_worldcover_codes_are_mapped_to_valid_classes(self):
        labels = np.array(WORLDCOVER_CODES, dtype=np.uint8)
        mapped = map_worldcover_labels(labels)

        self.assertEqual(mapped.shape, labels.shape)
        self.assertTrue(np.all(np.isin(mapped, np.arange(5, dtype=np.uint8))))
        self.assertEqual(set(mapped.tolist()), set(WORLDCOVER_TO_CLASS.values()))

    def test_documented_class_mapping(self):
        labels = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        expected = np.array([
            VEGETATION, VEGETATION, VEGETATION, VEGETATION, BUILT_UP,
            BARE_SOIL, OTHER, WATER, VEGETATION, VEGETATION, VEGETATION,
        ], dtype=np.uint8)
        np.testing.assert_array_equal(map_worldcover_labels(labels), expected)

    def test_unknown_codes_are_other(self):
        labels = np.array([-1, 0, 1, 255, 999], dtype=np.int16)
        expected = np.full(labels.shape, OTHER, dtype=np.uint8)
        np.testing.assert_array_equal(map_worldcover_labels(labels), expected)

    def test_class_ids_and_names_are_stable(self):
        self.assertEqual((OTHER, VEGETATION, WATER, BUILT_UP, BARE_SOIL), (0, 1, 2, 3, 4))
        self.assertEqual(CLASS_NAMES, ("Other", "Vegetation", "Water", "Built-up", "Bare Soil"))

if __name__ == "__main__":
    unittest.main()
