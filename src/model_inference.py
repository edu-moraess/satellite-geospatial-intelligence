from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    tile_x: int = 0
    tile_y: int = 0


class RemoteSensingInference:

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

    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        if image.ndim != 3:

            raise ValueError(
                "Expected RGB image with shape H x W x C."
            )

        if image.shape[2] != 3:

            raise ValueError(
                "Expected exactly 3 channels."
            )

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        minimum = image.min()
        maximum = image.max()

        if maximum > minimum:

            image = (
                image - minimum
            ) / (
                maximum - minimum
            )

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        return image

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.50,
    ) -> list[Detection]:

        """
        Executes model inference.

        IMPORTANT:
        No artificial detections are generated.

        When a trained checkpoint is connected,
        this method becomes the inference entry point.
        """

        image = self.preprocess(
            image
        )

        if self.model is None:

            return []

        # --------------------------------------------------
        # Model-specific inference
        # --------------------------------------------------

        predictions = self.model(
            image
        )

        detections = []

        for prediction in predictions:

            if prediction["confidence"] < confidence:
                continue

            detections.append(
                Detection(
                    label=prediction["label"],
                    confidence=float(
                        prediction["confidence"]
                    ),
                    x1=float(
                        prediction["x1"]
                    ),
                    y1=float(
                        prediction["y1"]
                    ),
                    x2=float(
                        prediction["x2"]
                    ),
                    y2=float(
                        prediction["y2"]
                    ),
                )
            )

        return detections

    def predict_tiles(
        self,
        tiles: list[dict],
        confidence: float = 0.50,
    ) -> list[Detection]:

        detections = []

        for tile in tiles:

            tile_image = tile["image"]

            tile_x = tile.get(
                "x",
                0,
            )

            tile_y = tile.get(
                "y",
                0,
            )

            tile_detections = self.predict(
                tile_image,
                confidence=confidence,
            )

            for detection in tile_detections:

                detection.tile_x = tile_x
                detection.tile_y = tile_y

                detection.x1 += tile_x
                detection.x2 += tile_x

                detection.y1 += tile_y
                detection.y2 += tile_y

                detections.append(
                    detection
                )

        return detections