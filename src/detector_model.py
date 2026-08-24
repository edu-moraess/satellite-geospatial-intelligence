from __future__ import annotations

from src.model_inference import (
    RemoteSensingInference,
)

from src.model_registry import (
    get_model,
    model_available,
)


class SatelliteDetector:

    def __init__(
        self,
        model_id: str = "remote_sensing_baseline",
        model=None,
        device: str = "cpu",
    ):

        self.model_id = model_id

        self.definition = get_model(
            model_id
        )

        self.engine = (
            RemoteSensingInference(
                model=model,
                device=device,
            )
        )

    @property
    def ready(self) -> bool:

        return self.engine.ready

    @property
    def checkpoint_available(self) -> bool:

        return model_available(
            self.model_id
        )

    def info(self) -> dict:

        return {

            "model_id": self.model_id,

            "model": self.definition.name,

            "description": (
                self.definition.description
            ),

            "backend": (
                "Remote Sensing Inference"
            ),

            "device": self.engine.device,

            "input_size": (
                self.definition.input_size
            ),

            "classes": (
                self.definition.classes
            ),

            "checkpoint_available": (
                self.checkpoint_available
            ),

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