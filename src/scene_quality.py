"""Scene-quality diagnostics for Sentinel-2 L2A observations.

The module is deliberately descriptive: it reports quality indicators from
the scene itself and never fabricates or silently repairs missing pixels.
"""
from __future__ import annotations

import numpy as np


SCL_VALID = (4, 5, 6, 7)
SCL_CLOUD = (8, 9, 10)
SCL_SHADOW = (2, 3)
SCL_SNOW = (11,)


def scene_quality_report(scl: np.ndarray | None, reference: np.ndarray) -> dict:
    """Return compact AOI-level quality metrics from the SCL layer."""
    if scl is None:
        return {
            "available": False,
            "valid_fraction": None,
            "cloud_fraction": None,
            "shadow_fraction": None,
            "snow_fraction": None,
            "score": None,
        }

    scl = np.asarray(scl)
    reference = np.asarray(reference)

    if scl.shape != reference.shape:
        raise ValueError(
            f"SCL/reference shape mismatch: {scl.shape} != {reference.shape}"
        )

    finite_reference = np.isfinite(reference)
    valid_pixels = finite_reference
    total = int(np.count_nonzero(valid_pixels))
    if total == 0:
        return {
            "available": True,
            "valid_fraction": 0.0,
            "cloud_fraction": None,
            "shadow_fraction": None,
            "snow_fraction": None,
            "score": 0.0,
        }

    values = scl[valid_pixels]
    cloud = np.isin(values, SCL_CLOUD)
    shadow = np.isin(values, SCL_SHADOW)
    snow = np.isin(values, SCL_SNOW)
    usable = np.isin(values, SCL_VALID)

    cloud_fraction = float(np.mean(cloud))
    shadow_fraction = float(np.mean(shadow))
    snow_fraction = float(np.mean(snow))
    usable_fraction = float(np.mean(usable))

    score = max(
        0.0,
        min(
            1.0,
            usable_fraction
            * (1.0 - cloud_fraction)
            * (1.0 - shadow_fraction)
            * (1.0 - snow_fraction),
        ),
    )

    return {
        "available": True,
        "valid_fraction": usable_fraction,
        "cloud_fraction": cloud_fraction,
        "shadow_fraction": shadow_fraction,
        "snow_fraction": snow_fraction,
        "score": score,
    }
