"""
Tests: src.object_detection
"""

import unittest

import numpy as np

from src.object_detection import (
    Detection,
    normalize_rgb,
    validate_detection_image,
    filter_detections,
    filter_classes,
    detection_summary,
)


class TestNormalizeRgb(unittest.TestCase):

    def test_output_shape_and_range(self):
        rng = np.random.default_rng(1)
        red = rng.uniform(0, 5000, size=(10, 10))
        green = rng.uniform(0, 5000, size=(10, 10))
        blue = rng.uniform(0, 5000, size=(10, 10))

        rgb = normalize_rgb(red, green, blue)

        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertTrue(np.all(rgb >= 0.0))
        self.assertTrue(np.all(rgb <= 1.0))

    def test_all_nan_channel_does_not_crash(self):
        nan_channel = np.full((5, 5), np.nan)
        ok_channel = np.ones((5, 5))

        rgb = normalize_rgb(nan_channel, ok_channel, ok_channel)
        self.assertEqual(rgb.shape, (5, 5, 3))


class TestValidateDetectionImage(unittest.TestCase):

    def test_valid_rgb_image_passes(self):
        image = np.ones((10, 10, 3), dtype=np.float32)
        self.assertTrue(validate_detection_image(image))

    def test_wrong_channels_raises(self):
        image = np.ones((10, 10, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_detection_image(image)

    def test_wrong_ndim_raises(self):
        image = np.ones((10, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_detection_image(image)

    def test_all_nan_raises(self):
        image = np.full((10, 10, 3), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_detection_image(image)


class TestFilters(unittest.TestCase):

    def _detections(self):
        return [
            Detection("building", 0.9, 0, 0, 10, 10),
            Detection("building", 0.4, 5, 5, 15, 15),
            Detection("vehicle", 0.6, 20, 20, 25, 25),
        ]

    def test_confidence_filter(self):
        result = filter_detections(self._detections(), 0.5)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(d.confidence >= 0.5 for d in result))

    def test_class_filter(self):
        result = filter_classes(self._detections(), ["vehicle"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "vehicle")

    def test_class_filter_empty_selection_returns_all(self):
        detections = self._detections()
        result = filter_classes(detections, [])
        self.assertEqual(len(result), len(detections))

    def test_detection_summary_counts(self):
        summary = detection_summary(self._detections())
        self.assertEqual(summary["building"], 2)
        self.assertEqual(summary["vehicle"], 1)

    def test_detection_summary_empty(self):
        self.assertEqual(detection_summary([]), {})


if __name__ == "__main__":
    unittest.main()
