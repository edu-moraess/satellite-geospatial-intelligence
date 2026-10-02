import numpy as np

from src.deep_learning.inference import build_inference_gate
from src.scene_quality import scene_quality_report


def test_scene_quality_masks_clouds_and_shadows():
    reference = np.ones((2, 4), dtype=np.float32)
    scl = np.array(
        [
            [4, 5, 6, 7],
            [8, 9, 3, 11],
        ],
        dtype=np.uint8,
    )

    report = scene_quality_report(scl, reference)

    assert report["available"] is True
    assert report["cloud_fraction"] == 0.25
    assert report["shadow_fraction"] == 0.125
    assert report["snow_fraction"] == 0.125
    assert report["valid_fraction"] == 0.5


def test_burn_scar_gate_blocks_unharmonized_sentinel_input():
    gate = build_inference_gate(
        "prithvi_eo_v2_300m_burn_scars",
        {"B02", "B03", "B04", "B8A", "B11", "B12"},
        source_domain="sentinel2-l2a",
    )

    assert gate.ready is False
    assert "harmonization" in gate.reason.lower()


def test_burn_scar_gate_reports_missing_bands_first():
    gate = build_inference_gate(
        "prithvi_eo_v2_300m_burn_scars",
        {"B02", "B03", "B04"},
        source_domain="sentinel2-l2a",
    )

    assert gate.ready is False
    assert set(gate.compatibility.missing_bands) == {"B12", "B11", "B8A"}
