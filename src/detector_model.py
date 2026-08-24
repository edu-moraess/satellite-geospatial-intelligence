from __future__ import annotations

from src.model_inference import (
    RemoteSensingInference,
)


class SatelliteDetector:

    def __init__(
        self,
        model_name: str = "Remote Sensing Detector",
        model=None,
        device: str = "cpu",
    ):

        self.model_name = model_name

        self.engine = (
            RemoteSensingInference(
                model=model,
                device=device,
            )
        )

    @property
    def ready(self) -> bool:

        return self.engine.ready

    def info(self) -> dict:

        return {
            "model": self.model_name,
            "backend": "Remote Sensing Inference",
            "device": self.engine.device,
            "ready": self.ready,
        }

    def predict_tiles(
        self,
        tiles,
        confidence: float = 0.50,
    ):

        return self.engine.predict_tiles(
            tiles,
            confidence=confidence,
        )