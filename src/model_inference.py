from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    """
    Standard detection object used by the
    geospatial computer vision pipeline.
    """

    label: str

    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    tile_x: int = 0
    tile_y: int = 0


class RemoteSensingInference:
    """
    Generic inference interface for remote-sensing models.

    This class intentionally does NOT fabricate detections.

    A real trained checkpoint must be connected through
    the predict() method.
    """

    def __init__(
        self,
        model: Any = None,
        device: str = "cpu",
    ):

        self.model = model
        self.device = device

    @property
    def ready(self) -> bool:

        return self.model is not None

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.50,
    ) -> list[Detection]:
        """
        Run inference on one image.

        Returns an empty list when no real model
        is connected.
        """

        if self.model is None:
            return []

        # ----------------------------------------------------
        # Model-specific inference will be implemented here.
        # ----------------------------------------------------

        return []

    def predict_tiles(
        self,
        tiles: list[dict],
        confidence: float = 0.50,
    ) -> list[Detection]:

        detections: list[Detection] = []

        for tile in tiles:

            image = tile["image"]

            tile_x = tile.get(
                "x",
                0,
            )

            tile_y = tile.get(
                "y",
                0,
            )

            tile_detections = self.predict(
                image=image,
                confidence=confidence,
            )

            for detection in tile_detections:

                detection.tile_x = tile_x
                detection.tile_y = tile_y

                detections.append(
                    detection
                )

        return detections