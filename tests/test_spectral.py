"""
Tests: src.spectral (NDVI / NDWI / NDBI)
"""

import unittest

import numpy as np

from src.spectral import (
    normalized_difference,
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
)


class TestNormalizedDifference(unittest.TestCase):

    def test_known_values(self):
        a = np.array([[8.0]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)

        # (8-2)/(8+2) = 0.6
        result = normalized_difference(a, b)
        self.assertAlmostEqual(float(result[0, 0]), 0.6, places=5)

    def test_result_is_bounded(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 10000, size=(20, 20)).astype(np.float32)
        b = rng.uniform(0, 10000, size=(20, 20)).astype(np.float32)

        result = normalized_difference(a, b)
        valid = result[np.isfinite(result)]

        self.assertTrue(np.all(valid >= -1.0001))
        self.assertTrue(np.all(valid <= 1.0001))

    def test_zero_denominator_becomes_nan_not_crash(self):
        a = np.array([[0.0]], dtype=np.float32)
        b = np.array([[0.0]], dtype=np.float32)

        result = normalized_difference(a, b)
        self.assertTrue(np.isnan(result[0, 0]))

    def test_shape_mismatch_raises(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)

        with self.assertRaises(ValueError):
            normalized_difference(a, b)


class TestNdviNdwiNdbi(unittest.TestCase):

    def test_ndvi_high_for_vegetation_like_values(self):
        # Vegetation: high NIR, low RED.
        red = np.array([[500.0]], dtype=np.float32)
        nir = np.array([[4000.0]], dtype=np.float32)

        ndvi = calculate_ndvi(red, nir)
        self.assertGreater(float(ndvi[0, 0]), 0.5)

    def test_ndwi_high_for_water_like_values(self):
        # Water: high GREEN, low NIR.
        green = np.array([[3000.0]], dtype=np.float32)
        nir = np.array([[300.0]], dtype=np.float32)

        ndwi = calculate_ndwi(green, nir)
        self.assertGreater(float(ndwi[0, 0]), 0.5)

    def test_ndbi_high_for_built_up_like_values(self):
        # Built-up: high SWIR, low NIR.
        nir = np.array([[500.0]], dtype=np.float32)
        swir = np.array([[3500.0]], dtype=np.float32)

        ndbi = calculate_ndbi(nir, swir)
        self.assertGreater(float(ndbi[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
