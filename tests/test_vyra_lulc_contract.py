"""Tests for the verified VYRA B07 field-test contract."""

import numpy as np
import pytest

from src.vyra_lulc import (
    CONTRACT,
    EXPECTED_CHECKPOINT_SHA256,
    PATCH_FINGERPRINT,
    prepare_b07_patch,
)


def test_contract_is_frozen():
    assert CONTRACT.input_shape == (1, 64, 64)
    assert CONTRACT.classes == 29
    assert CONTRACT.band == "B07"
    assert CONTRACT.stac_asset_key == "rededge3"
    assert CONTRACT.patch_size == 64
    assert CONTRACT.stride == 64
    assert CONTRACT.gsd_m == 20
    assert CONTRACT.patch_size_m == 1280
    assert CONTRACT.augmentation == "none"
    assert CONTRACT.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert CONTRACT.patch_fingerprint == PATCH_FINGERPRINT


def test_b07_patch_preparation_preserves_values_and_shape():
    source = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    result = prepare_b07_patch(source)

    assert result.shape == (1, 64, 64)
    np.testing.assert_array_equal(result[0], source)


@pytest.mark.parametrize(
    "shape",
    [(64, 63), (63, 64), (1, 64, 64)],
)
def test_b07_patch_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        prepare_b07_patch(np.zeros(shape, dtype=np.float32))
