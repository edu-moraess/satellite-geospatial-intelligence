"""
Satellite Object Detector
=========================

Model abstraction layer.

The application does not directly depend
on a specific AI model implementation.

This allows us to plug in:
- remote sensing models
- YOLO variants
- GeoAI models
- custom trained models
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelDetection:

    label: str

    confidence: float

    x1: float
    y1: float

    x2: float
    y2: float


class SatelliteDetector:
    """
    Generic satellite detector interface.
    """

    def __init__(
        self,
        model_name: str = "remote-sensing-detector",
    ):

        self.model_name = model_name

        self.loaded = False


    def load(self):
        """
        Load the model.

        The actual model implementation
        will be connected in the next step.
        """

        self.loaded = True

        return True


    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.50,
    ):
        """
        Run object detection.

        Currently returns an empty list because
        the specialized model has not yet been
        connected.
        """

        if not self.loaded:

            self.load()

        image = np.asarray(
            image
        )

        if image.ndim != 3:

            raise ValueError(
                "Input image must have "
                "three dimensions."
            )

        if image.shape[-1] != 3:

            raise ValueError(
                "Input image must have "
                "three channels."
            )

        return []


    def info(self):

        return {
            "model": self.model_name,
            "loaded": self.loaded,
            "status": (
                "ready for model integration"
            ),
        }