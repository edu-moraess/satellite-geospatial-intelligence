"""
Tests: src.change_detection
"""

import unittest

import numpy as np

from src.change_detection import (
    calculate_difference,
    detect_change,
    calculate_change_statistics,
    normalized_change,
)
from src.raster_validation import RasterValidationError


class TestCalculateDifference(unittest.TestCase):

    def test_basic_difference(self):
        before = np.array([[0.2]], dtype=np.float32)
        after = np.array([[0.5]], dtype=np.float32)

        diff = calculate_difference(before, after)
        self.assertAlmostEqual(float(diff[0, 0]), 0.3, places=5)

    def test_shape_mismatch_raises(self):
        before = np.zeros((4, 4), dtype=np.float32)
        after = np.zeros((5, 5), dtype=np.float32)

        with self.assertRaises(RasterValidationError):
            calculate_difference(before, after)

    def test_crs_mismatch_raises_when_metadata_given(self):
        before = np.zeros((3, 3), dtype=np.float32)
        after = np.zeros((3, 3), dtype=np.float32)

        with self.assertRaises(RasterValidationError):
            calculate_difference(
                before, after,
                before_metadata={"crs": "EPSG:32723", "transform": (1,0,0,0,-1,0)},
                after_metadata={"crs": "EPSG:32633", "transform": (1,0,0,0,-1,0)},
            )

    def test_no_metadata_still_works(self):
        # Backward compatible: bare arrays, no metadata.
        before = np.array([[0.1, 0.2]], dtype=np.float32)
        after = np.array([[0.3, 0.1]], dtype=np.float32)

        diff = calculate_difference(before, after)
        self.assertEqual(diff.shape, (1, 2))


class TestDetectChange(unittest.TestCase):

    def test_increase_decrease_and_stable(self):
        difference = np.array(
            [[0.5, -0.5, 0.01]], dtype=np.float32
        )

        result = detect_change(difference, threshold=0.1)

        self.assertEqual(result[0, 0], 1)   # increase
        self.assertEqual(result[0, 1], -1)  # decrease
        self.assertEqual(result[0, 2], 0)   # stable


class TestChangeStatistics(unittest.TestCase):

    def test_statistics_counts_and_areas(self):
        # 10m pixels -> 100 m^2 -> 0.0001 km^2 per pixel.
        change_map = np.array(
            [[1, 1, -1, 0]], dtype=np.int8
        )

        stats = calculate_change_statistics(
            change_map, pixel_size_meters=10.0
        )

        self.assertEqual(stats["increase_pixels"], 2)
        self.assertEqual(stats["decrease_pixels"], 1)
        self.assertEqual(stats["unchanged_pixels"], 1)
        self.assertAlmostEqual(
            stats["total_changed_km2"],
            3 * (10.0 ** 2) / 1_000_000.0,
            places=8,
        )


class TestNormalizedChange(unittest.TestCase):

    def test_relative_change(self):
        before = np.array([[1.0]], dtype=np.float32)
        after = np.array([[1.5]], dtype=np.float32)

        result = normalized_change(before, after)
        self.assertAlmostEqual(float(result[0, 0]), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
