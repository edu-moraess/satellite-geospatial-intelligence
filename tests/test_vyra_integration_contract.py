"""Regression tests for the VYRA SmallCNN satellite integration contract."""

from src.model_registry import get_model, model_available
from src.sensor_registry import SENSORS


def test_vyra_registry_identity_is_frozen():
    model = get_model("vyra_smallcnn_b07_lulc_v1")

    assert model.task == "lulc_classification"
    assert model.band == "B07"
    assert model.channels == 1
    assert model.input_size == 64
    assert len(model.classes) == 29
    assert model.checkpoint_sha256 == (
        "ef56a68ad7362d4bb1604b29c5669ac27851284bd15cb56468a036b1c4d90c2a"
    )
    assert model.patch_fingerprint == (
        "e5ad2d547726a22a8a069247946f099be293c7d7f6488924551c927b240b2d72"
    )
    assert model.experiment_fingerprint == (
        "beaecb493ad09e6cb60740447053dffa20c4019e904cb5906c335252b7b1332f"
    )
    assert model.status == "BLOCKED_UNTIL_CHECKPOINT_CONTRACT"
    assert model_available("vyra_smallcnn_b07_lulc_v1") is False


def test_sentinel2_registry_exposes_vyra_b07():
    assert SENSORS["sentinel2"].bands["rededge3"] == "B07"


def test_vyra_class_contract_matches_canonical_ids():
    model = get_model("vyra_smallcnn_b07_lulc_v1")
    assert model.classes[0] == "BarrenLands__"
    assert model.classes[-1] == "UrbanBlUpArea"
