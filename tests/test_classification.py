"""
Tests: src.classification (rule-based land cover)
"""

import unittest

import numpy as np

from src.classification import (
    classify_land_cover,
    WATER,
    VEGETATION,
    BUILT_UP,
    OTHER,
)


class TestClassifyLandCover(unittest.TestCase):

    def test_water_pixel(self):
        ndvi = np.array([[0.1]], dtype=np.float32)
        ndwi = np.array([[0.5]], dtype=np.float32)
        ndbi = np.array([[-0.2]], dtype=np.float32)

        result = classify_land_cover(ndvi, ndwi, ndbi)
        self.assertEqual(result[0, 0], WATER)

    def test_vegetation_pixel(self):
        ndvi = np.array([[0.7]], dtype=np.float32)
        ndwi = np.array([[-0.3]], dtype=np.float32)
        ndbi = np.array([[-0.2]], dtype=np.float32)

        result = classify_land_cover(ndvi, ndwi, ndbi)
        self.assertEqual(result[0, 0], VEGETATION)

    def test_built_up_pixel(self):
        ndvi = np.array([[0.05]], dtype=np.float32)
        ndwi = np.array([[-0.3]], dtype=np.float32)
        ndbi = np.array([[0.30]], dtype=np.float32)

        result = classify_land_cover(ndvi, ndwi, ndbi)
        self.assertEqual(result[0, 0], BUILT_UP)

    def test_nan_pixel_is_other(self):
        ndvi = np.array([[np.nan]], dtype=np.float32)
        ndwi = np.array([[0.5]], dtype=np.float32)
        ndbi = np.array([[0.1]], dtype=np.float32)

        result = classify_land_cover(ndvi, ndwi, ndbi)
        self.assertEqual(result[0, 0], OTHER)

    def test_shape_mismatch_raises(self):
        ndvi = np.zeros((4, 4), dtype=np.float32)
        ndwi = np.zeros((5, 5), dtype=np.float32)
        ndbi = np.zeros((4, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            classify_land_cover(ndvi, ndwi, ndbi)

    def test_output_dtype_is_uint8(self):
        ndvi = np.zeros((3, 3), dtype=np.float32)
        ndwi = np.zeros((3, 3), dtype=np.float32)
        ndbi = np.zeros((3, 3), dtype=np.float32)

        result = classify_land_cover(ndvi, ndwi, ndbi)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
