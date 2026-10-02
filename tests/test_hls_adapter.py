import numpy as np
from rasterio.transform import from_origin

from src.deep_learning.hls_adapter import (
    PRITHVI_BANDS,
    PRITHVI_MEANS,
    PRITHVI_STDS,
    build_prithvi_input,
)
from src.deep_learning.inference import build_inference_gate


def _metadata(pixel_size: float) -> dict:
    width = 6
    height = 6
    transform = from_origin(0, 180, pixel_size, pixel_size)
    return {
        "transform": transform,
        "crs": "EPSG:32623",
        "width": width,
        "height": height,
        "bounds": (
            type("Bounds", (), {
                "left": 0.0, "bottom": 0.0,
                "right": width * pixel_size, "top": height * pixel_size,
            })()
        ),
    }


def test_prithvi_adapter_builds_six_band_30m_cube():
    bands = {band: np.ones((6, 6), dtype=np.float32) * 0.1 for band in PRITHVI_BANDS}
    metadata = {band: _metadata(10.0 if band in {"B02", "B03", "B04"} else 20.0)
                for band in PRITHVI_BANDS}

    result = build_prithvi_input(bands, metadata)

    assert result.cube.shape == (6, 2, 2)
    assert result.bands == PRITHVI_BANDS
    assert result.resolution_m == 30.0
    assert result.harmonization_status == "spatially_harmonized"
    assert np.allclose(
        result.cube[:, 0, 0],
        (0.1 - PRITHVI_MEANS) / PRITHVI_STDS,
    )


def test_prithvi_gate_does_not_accept_spatial_only_adapter():
    gate = build_inference_gate(
        "prithvi_eo_v2_300m_burn_scars",
        set(PRITHVI_BANDS),
        source_domain="sentinel2-l2a",
        harmonization_status="spatially_harmonized",
    )

    assert gate.ready is False


def test_prithvi_gate_accepts_only_explicit_hls_status_before_checkpoint_check():
    gate = build_inference_gate(
        "prithvi_eo_v2_300m_burn_scars",
        set(PRITHVI_BANDS),
        source_domain="sentinel2-l2a",
        harmonization_status="hls_s30",
    )

    assert gate.ready is False
    assert "checkpoint" in gate.reason.lower()
