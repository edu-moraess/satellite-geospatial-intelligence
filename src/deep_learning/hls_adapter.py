"""Sentinel-2 to Prithvi/HLS-compatible multispectral adapter.

Builds the six-band 30 m tensor required by the Prithvi-EO-2.0 Burn Scars
checkpoint and applies the checkpoint-published normalization.

This is intentionally not a claim of full NASA HLS S30 reproduction.
HLS also performs spectral bandpass adjustment and nadir/BRDF normalization.
The result is therefore marked as spatially_harmonized and remains blocked
from Burn Scars inference until those sensor-domain adjustments are validated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from src.deep_learning.inference import validate_cube


PRITHVI_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")

PRITHVI_MEANS = np.asarray([
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
], dtype=np.float32)

PRITHVI_STDS = np.asarray([
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
], dtype=np.float32)


@dataclass(frozen=True)
class HLSAdapterResult:
    """Model-ready tensor with explicit provenance metadata."""

    cube: np.ndarray
    transform: object
    crs: object
    resolution_m: float
    bands: tuple[str, ...]
    normalization: str
    harmonization_status: str
    warning: str


def _target_grid(reference_metadata: dict, resolution_m: float) -> tuple[object, int, int]:
    bounds = reference_metadata["bounds"]
    width = max(1, int(np.ceil((bounds.right - bounds.left) / resolution_m)))
    height = max(1, int(np.ceil((bounds.top - bounds.bottom) / resolution_m)))
    transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
    )
    return transform, width, height


def _resample_band(array: np.ndarray, metadata: dict, transform: object, width: int, height: int) -> np.ndarray:
    source = np.asarray(array, dtype=np.float32)
    destination = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=source, destination=destination,
        src_transform=metadata["transform"], src_crs=metadata["crs"],
        dst_transform=transform, dst_crs=metadata["crs"],
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return destination


def build_prithvi_input(
    bands: dict[str, np.ndarray],
    metadata: dict[str, dict],
    resolution_m: float = 30.0,
    normalize: bool = True,
) -> HLSAdapterResult:
    """Build the six-band 30 m Prithvi input from Sentinel-2 reflectance."""
    missing = [band for band in PRITHVI_BANDS if band not in bands or band not in metadata]
    if missing:
        raise ValueError("Prithvi input cannot be built; missing Sentinel-2 bands: " + ", ".join(missing))

    reference = metadata["B04"]
    transform, width, height = _target_grid(reference, resolution_m)
    aligned = [
        _resample_band(bands[band], metadata[band], transform, width, height)
        for band in PRITHVI_BANDS
    ]
    cube = np.stack(aligned, axis=0).astype(np.float32, copy=False)
    if normalize:
        cube = (cube - PRITHVI_MEANS[:, None, None]) / PRITHVI_STDS[:, None, None]
    cube = validate_cube(cube, expected_bands=6)
    return HLSAdapterResult(
        cube=cube, transform=transform, crs=reference["crs"],
        resolution_m=float(resolution_m), bands=PRITHVI_BANDS,
        normalization="Prithvi Burn Scars mean/std" if normalize else "surface reflectance",
        harmonization_status="spatially_harmonized",
        warning=("Not HLS S30: spectral bandpass and BRDF/nadir normalization are not applied. "
                 "Burn Scars inference must remain gated."),
    )
