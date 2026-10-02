import unittest

import numpy as np

from src.ml.features import FEATURE_ORDER, extract_features


def _bands(shape=(2, 2)):
    return {
        "B02": np.full(shape, 1000.0, dtype=np.float32),
        "B03": np.full(shape, 1200.0, dtype=np.float32),
        "B04": np.full(shape, 1500.0, dtype=np.float32),
        "B08": np.full(shape, 4000.0, dtype=np.float32),
        "B11": np.full(shape, 3500.0, dtype=np.float32),
    }


class TestFeatureExtraction(unittest.TestCase):
    def test_output_shape_and_order(self):
        result = extract_features(_bands())
        self.assertEqual(result.matrix.shape, (4, 8))
        self.assertEqual(result.feature_names, FEATURE_ORDER)
        self.assertEqual(int(result.valid_mask.sum()), 4)

    def test_nan_pixels_are_excluded(self):
        bands = _bands()
        bands["B04"][0, 1] = np.nan
        result = extract_features(bands)
        self.assertEqual(result.matrix.shape, (3, 8))
        self.assertFalse(result.valid_mask[0, 1])
        self.assertTrue(np.all(np.isfinite(result.matrix)))

    def test_explicit_scaling_is_applied_and_preserved(self):
        scaling = {
            name: {"scale": 0.0001, "offset": 0.0}
            for name in ("B02", "B03", "B04", "B08", "B11")
        }
        result = extract_features(_bands(), reflectance_scaling=scaling)
        self.assertAlmostEqual(float(result.matrix[0, 0]), 0.1, places=6)
        self.assertEqual(result.reflectance_scaling["B02"]["scale"], 0.0001)

    def test_missing_band_scaling_is_rejected(self):
        scaling = {
            name: {"scale": 0.0001, "offset": 0.0}
            for name in ("B02", "B03", "B04", "B08")
        }
        with self.assertRaises(ValueError):
            extract_features(_bands(), reflectance_scaling=scaling)

    def test_missing_scale_field_is_rejected(self):
        scaling = {
            name: {"scale": 0.0001, "offset": 0.0}
            for name in ("B02", "B03", "B04", "B08", "B11")
        }
        del scaling["B11"]["scale"]
        with self.assertRaises(ValueError):
            extract_features(_bands(), reflectance_scaling=scaling)

    def test_none_means_no_transformation(self):
        result = extract_features(_bands(), reflectance_scaling=None)
        self.assertIsNone(result.reflectance_scaling)
        self.assertAlmostEqual(float(result.matrix[0, 0]), 1000.0, places=5)


if __name__ == "__main__":
    unittest.main()
