"""
Tests: src.raster_validation

These are the guardrails behind the "band_a + band_b"
ValueError bug and the general Raster Alignment / Validation
layer requested for the platform.
"""

import unittest

import numpy as np

from src.raster_validation import (
    RasterValidationError,
    validate_raster,
    validate_raster_pair,
)


def _band(shape=(4, 4), fill=1.0, nan_at=None):
    array = np.full(shape, fill, dtype=np.float32)
    if nan_at is not None:
        array[nan_at] = np.nan
    return array


def _meta(crs="EPSG:32723", transform=(10, 0, 500000, 0, -10, 8000000)):
    return {"crs": crs, "transform": transform}


class TestValidateSingleRaster(unittest.TestCase):

    def test_valid_raster_passes(self):
        info = validate_raster(_band(), label="test")
        self.assertEqual(info["shape"], (4, 4))

    def test_none_raster_raises(self):
        with self.assertRaises(RasterValidationError):
            validate_raster(None, label="missing band")

    def test_empty_raster_raises(self):
        with self.assertRaises(RasterValidationError):
            validate_raster(np.array([]), label="empty band")

    def test_wrong_ndim_raises(self):
        with self.assertRaises(RasterValidationError):
            validate_raster(np.zeros((2, 2, 3)), label="3d band")

    def test_all_nan_raster_raises(self):
        array = np.full((4, 4), np.nan, dtype=np.float32)
        with self.assertRaises(RasterValidationError):
            validate_raster(array, label="all nan band")

    def test_min_valid_fraction_enforced(self):
        array = _band((10, 10))
        array[:9, :] = np.nan  # only 10% valid

        with self.assertRaises(RasterValidationError):
            validate_raster(
                array, label="mostly nan", min_valid_fraction=0.5
            )


class TestValidateRasterPair(unittest.TestCase):

    def test_matching_pair_passes(self):
        a = _band(fill=1.0)
        b = _band(fill=2.0)

        info = validate_raster_pair(a, b)
        self.assertEqual(info["shape"], (4, 4))

    def test_shape_mismatch_raises(self):
        a = _band((4, 4))
        b = _band((5, 5))

        with self.assertRaises(RasterValidationError):
            validate_raster_pair(a, b)

    def test_crs_mismatch_raises(self):
        a = _band()
        b = _band()

        with self.assertRaises(RasterValidationError):
            validate_raster_pair(
                a, b,
                metadata_a=_meta(crs="EPSG:32723"),
                metadata_b=_meta(crs="EPSG:32633"),
            )

    def test_transform_mismatch_raises(self):
        a = _band()
        b = _band()

        with self.assertRaises(RasterValidationError):
            validate_raster_pair(
                a, b,
                metadata_a=_meta(transform=(10, 0, 500000, 0, -10, 8000000)),
                metadata_b=_meta(transform=(10, 0, 999999, 0, -10, 8000000)),
            )

    def test_matching_crs_and_transform_passes(self):
        a = _band()
        b = _band()
        meta = _meta()

        # Should not raise.
        validate_raster_pair(
            a, b, metadata_a=meta, metadata_b=meta
        )

    def test_no_overlap_raises(self):
        a = _band(nan_at=slice(None))  # fully NaN
        b = _band()

        with self.assertRaises(RasterValidationError):
            validate_raster_pair(a, b)


if __name__ == "__main__":
    unittest.main()
