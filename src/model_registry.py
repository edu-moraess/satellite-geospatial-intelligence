from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    description: str
    checkpoint: Path | None
    classes: tuple[str, ...]
    input_size: int
    task: str = "object_detection"
    band: str | None = None
    channels: int = 3
    checkpoint_sha256: str | None = None
    patch_fingerprint: str | None = None
    experiment_fingerprint: str | None = None
    status: str = "UNAVAILABLE"


VYRA_LULC_CLASSES = (
    "BarrenLands__",
    "MossAndLichen",
    "Grasslands___",
    "ShrublandOpen",
    "SrublandClose",
    "ForestsOpDeBr",
    "ForestsClDeBr",
    "ForestsDeDeBr",
    "ForestsOpDeNe",
    "ForestsClDeNe",
    "ForestsDeDeNe",
    "ForestsOpEvBr",
    "ForestsClEvBr",
    "ForestsDeEvBr",
    "ForestsOpEvNe",
    "ForestsClEvNe",
    "ForestsDeEvNe",
    "WetlandMangro",
    "WetlandSwamps",
    "WetlandMarshl",
    "WaterBodyMari",
    "WaterBodyCont",
    "PermanentSnow",
    "CropSeasWater",
    "CropCereaIrri",
    "CropCereaRain",
    "CropBroadIrri",
    "CropBroadRain",
    "UrbanBlUpArea",
)


AVAILABLE_MODELS = {
    "remote_sensing_baseline": ModelDefinition(
        name="Remote Sensing Baseline",
        description=(
            "Interface preparada para um "
            "checkpoint específico de sensoriamento remoto."
        ),
        checkpoint=None,
        classes=(
            "Buildings",
            "Roads",
            "Vehicles",
            "Aircraft",
            "Ships",
            "Storage Tanks",
        ),
        input_size=512,
    ),
    "vyra_smallcnn_b07_lulc_v1": ModelDefinition(
        name="VYRA SmallCNN — Sentinel-2 B07 LULC",
        description=(
            "Frozen VYRA GATE 8 baseline for 64×64 single-band "
            "Sentinel-2 B07 / rededge3 land-cover classification."
        ),
        checkpoint=None,
        classes=VYRA_LULC_CLASSES,
        input_size=64,
        task="lulc_classification",
        band="B07",
        channels=1,
        checkpoint_sha256=(
            "ef56a68ad7362d4bb1604b29c5669ac27851284bd15cb56468a036b1c4d90c2a"
        ),
        patch_fingerprint=(
            "e5ad2d547726a22a8a069247946f099be293c7d7f6488924551c927b240b2d72"
        ),
        experiment_fingerprint=(
            "beaecb493ad09e6cb60740447053dffa20c4019e904cb5906c335252b7b1332f"
        ),
        status="BLOCKED_UNTIL_CHECKPOINT_CONTRACT",
    ),
}


def get_model(model_id: str) -> ModelDefinition:
    if model_id not in AVAILABLE_MODELS:
        raise KeyError(f"Unknown model: {model_id}")
    return AVAILABLE_MODELS[model_id]


def list_models() -> list[str]:
    return list(AVAILABLE_MODELS.keys())


def model_available(model_id: str) -> bool:
    model = get_model(model_id)
    if model.checkpoint is None:
        return False
    return model.checkpoint.exists()
