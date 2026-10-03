"""Feature extraction for experimental ML land-cover classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.spectral import calculate_ndbi, calculate_ndvi, calculate_ndwi

BAND_ORDER = ("B02", "B03", "B04", "B08", "B11")
FEATURE_ORDER = ("B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "NDBI")

# Verified from an Earth Search Sentinel-2 L2A scene (S2A_23KLQ_20211107_1_L2A):
# raster:bands reports scale=0.0001 and offset=-0.1 for all five ML bands.
SENTINEL2_L2A_REFLECTANCE_SCALING = {
    name: {"scale": 0.0001, "offset": -0.1} for name in BAND_ORDER
}


@dataclass(frozen=True)
class FeatureResult:
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    valid_mask: np.ndarray
    reflectance_scaling: dict[str, dict[str, float]] | None


def _validate_bands(bands: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in BAND_ORDER if name not in bands]
    if missing:
        raise ValueError(f"Missing required Sentinel-2 bands: {', '.join(missing)}")
    shapes = {np.asarray(bands[name]).shape for name in BAND_ORDER}
    if len(shapes) != 1:
        raise ValueError(
            "All required bands must have identical aligned shapes. "
            f"Shapes found: {sorted(shapes)}"
        )
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(f"Expected 2-D raster bands, got shape {shape!r}")


def _apply_scaling(
    bands: Mapping[str, np.ndarray],
    reflectance_scaling: Mapping[str, Mapping[str, float]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]] | None]:
    arrays = {name: np.asarray(bands[name], dtype=np.float32) for name in BAND_ORDER}
    if reflectance_scaling is None:
        return arrays, None

    missing = [name for name in BAND_ORDER if name not in reflectance_scaling]
    if missing:
        raise ValueError(
            "reflectance_scaling is missing required bands: " + ", ".join(missing)
        )

    scaled: dict[str, np.ndarray] = {}
    used: dict[str, dict[str, float]] = {}
    for name in BAND_ORDER:
        config = reflectance_scaling[name]
        if "scale" not in config:
            raise ValueError(f"reflectance_scaling[{name!r}] must define 'scale'")
        scale = float(config["scale"])
        offset = float(config.get("offset", 0.0))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError(
                f"reflectance_scaling[{name!r}]['scale'] must be finite and non-zero"
            )
        if not np.isfinite(offset):
            raise ValueError(f"reflectance_scaling[{name!r}]['offset'] must be finite")
        scaled[name] = arrays[name] * scale + offset
        used[name] = {"scale": scale, "offset": offset}
    return scaled, used


def extract_features(
    bands: Mapping[str, np.ndarray],
    *,
    reflectance_scaling: Mapping[str, Mapping[str, float]] | None = None,
    input_valid_mask: np.ndarray | None = None,
) -> FeatureResult:
    """Build an ML-ready feature matrix from aligned Sentinel-2 bands.

    ``None`` is an explicit no-transformation choice. No scaling convention is inferred.
    For Earth Search Sentinel-2 L2A COGs, pass
    ``SENTINEL2_L2A_REFLECTANCE_SCALING`` explicitly.
    """
    _validate_bands(bands)
    if input_valid_mask is not None:
        input_valid_mask = np.asarray(input_valid_mask, dtype=bool)
        expected_shape = np.asarray(bands[BAND_ORDER[0]]).shape
        if input_valid_mask.shape != expected_shape:
            raise ValueError(
                "input_valid_mask must match the aligned band shape: "
                f"{input_valid_mask.shape} != {expected_shape}"
            )
    bands, scaling_used = _apply_scaling(bands, reflectance_scaling)

    ndvi = calculate_ndvi(bands["B04"], bands["B08"])
    ndwi = calculate_ndwi(bands["B03"], bands["B08"])
    ndbi = calculate_ndbi(bands["B08"], bands["B11"])

    cube = np.stack(
        [
            bands["B02"], bands["B03"], bands["B04"], bands["B08"], bands["B11"],
            ndvi, ndwi, ndbi,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    valid_mask = np.all(np.isfinite(cube), axis=-1)
    if input_valid_mask is not None:
        valid_mask &= input_valid_mask
    return FeatureResult(
        matrix=cube[valid_mask],
        feature_names=FEATURE_ORDER,
        valid_mask=valid_mask,
        reflectance_scaling=scaling_used,
    )
