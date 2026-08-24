"""
Satellite Object Detection
==========================

Prepares satellite imagery for object detection.

Important:
Sentinel-2 has relatively coarse spatial resolution.
Object detection here is a baseline pipeline.

The module prepares RGB imagery and converts
detections into geographic-aware metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# DETECTION
# ============================================================

@dataclass
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_rgb(
    red,
    green,
    blue,
):
    """
    Normalize Sentinel-2 reflectance bands
    into an RGB image suitable for visualization
    and model inference.
    """

    red = np.asarray(
        red,
        dtype=np.float32,
    )

    green = np.asarray(
        green,
        dtype=np.float32,
    )

    blue = np.asarray(
        blue,
        dtype=np.float32,
    )


    def stretch(channel):

        valid = channel[
            np.isfinite(channel)
        ]

        if valid.size == 0:

            return np.zeros_like(
                channel
            )

        low = np.percentile(
            valid,
            2,
        )

        high = np.percentile(
            valid,
            98,
        )

        if high <= low:

            return np.zeros_like(
                channel
            )

        result = (
            channel - low
        ) / (
            high - low
        )

        result = np.clip(
            result,
            0,
            1,
        )

        return result


    rgb = np.stack(
        [
            stretch(red),
            stretch(green),
            stretch(blue),
        ],
        axis=-1,
    )


    return rgb


# ============================================================
# VALID IMAGE CHECK
# ============================================================

def validate_detection_image(
    image,
):
    """
    Validate image before model inference.
    """

    image = np.asarray(
        image
    )


    if image.ndim != 3:

        raise ValueError(
            "Detection image must "
            "have 3 dimensions."
        )


    if image.shape[-1] != 3:

        raise ValueError(
            "Detection image must "
            "contain exactly 3 channels."
        )


    if image.size == 0:

        raise ValueError(
            "Detection image is empty."
        )


    return True