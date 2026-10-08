"""VYRA GEOCORE SmallCNN inference adapter for the Satellite application.

No training logic is included. The adapter loads the frozen GATE 8
SmallCNN baseline and exposes 64x64 Sentinel-2 B07 LULC inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn

VYRA_CHECKPOINT_SHA256 = "ef56a68ad7362d4bb1604b29c5669ac27851284bd15cb56468a036b1c4d90c2a"
VYRA_PATCH_FINGERPRINT = "e5ad2d547726a22a8a069247946f099be293c7d7f6488924551c927b240b2d72"
VYRA_EXPERIMENT_FINGERPRINT = "beaecb493ad09e6cb60740447053dffa20c4019e904cb5906c335252b7b1332f"
INPUT_SIZE = 64
NUM_CLASSES = 29
EXPECTED_PARAMETERS = 25181

CLASS_NAMES = (
    "BarrenLands__", "MossAndLichen", "Grasslands___", "ShrublandOpen",
    "SrublandClose", "ForestsOpDeBr", "ForestsClDeBr", "ForestsDeDeBr",
    "ForestsOpDeNe", "ForestsClDeNe", "ForestsDeDeNe", "ForestsOpEvBr",
    "ForestsClEvBr", "ForestsDeEvBr", "ForestsOpEvNe", "ForestsClEvNe",
    "ForestsDeEvNe", "WetlandMangro", "WetlandSwamps", "WetlandMarshl",
    "WaterBodyMari", "WaterBodyCont", "PermanentSnow", "CropSeasWater",
    "CropCereaIrri", "CropCereaRain", "CropBroadIrri", "CropBroadRain",
    "UrbanBlUpArea",
)

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_name: str
    probability: float

def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_model(checkpoint: Path, device: str = "cpu") -> SmallCNN:
    actual = checkpoint_sha256(checkpoint)
    if actual != VYRA_CHECKPOINT_SHA256:
        raise ValueError(
            f"VYRA checkpoint SHA256 mismatch: expected {VYRA_CHECKPOINT_SHA256}, got {actual}"
        )
    model = SmallCNN()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(f"SmallCNN parameter mismatch: expected {EXPECTED_PARAMETERS}, got {parameters}")
    return model

def prepare_b07_patch(patch: np.ndarray) -> torch.Tensor:
    array = np.asarray(patch, dtype=np.float32)
    if array.shape != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"Expected B07 patch {(INPUT_SIZE, INPUT_SIZE)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("B07 patch contains NaN or Inf")
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

def predict_patch(model: SmallCNN, patch: np.ndarray, device: str = "cpu") -> Prediction:
    tensor = prepare_b07_patch(patch).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    class_id = int(torch.argmax(probabilities).item())
    probability = float(probabilities[class_id].item())
    return Prediction(class_id, CLASS_NAMES[class_id], probability)

def model_info(checkpoint: Path) -> dict:
    return {
        "model_id": "vyra_smallcnn_b07_lulc_v1",
        "name": "VYRA SmallCNN — Sentinel-2 B07 LULC",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": VYRA_CHECKPOINT_SHA256,
        "patch_fingerprint": VYRA_PATCH_FINGERPRINT,
        "experiment_fingerprint": VYRA_EXPERIMENT_FINGERPRINT,
        "input": "1x64x64",
        "band": "B07 / rededge3",
        "classes": NUM_CLASSES,
        "parameters": EXPECTED_PARAMETERS,
        "training_status": "frozen baseline",
    }
}
