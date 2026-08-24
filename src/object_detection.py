"""
Object Detection Engine
=======================

Satellite Geospatial Intelligence

Stage 4C:
- Detection engine
- Bounding boxes
- Confidence filtering
- Visualization preparation

The detector is intentionally separated from
the satellite ingestion pipeline.

This allows us to plug in a specialized
geospatial model later without changing
the rest of the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Detection:
    """
    Single detected object.
    """

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
    Convert Sentinel-2 bands into
    normalized RGB image.
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
                channel,
                dtype=np.float32,
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
                channel,
                dtype=np.float32,
            )

        result = (
            channel - low
        ) / (
            high - low
        )

        return np.clip(
            result,
            0.0,
            1.0,
        )

    return np.stack(
        [
            stretch(red),
            stretch(green),
            stretch(blue),
        ],
        axis=-1,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_detection_image(
    image,
):
    """
    Validate detection input.
    """

    image = np.asarray(
        image
    )

    if image.ndim != 3:

        raise ValueError(
            "Detection image must have "
            "three dimensions."
        )

    if image.shape[-1] != 3:

        raise ValueError(
            "Detection image must contain "
            "exactly three channels."
        )

    if image.size == 0:

        raise ValueError(
            "Detection image is empty."
        )

    if not np.isfinite(
        image
    ).any():

        raise ValueError(
            "Detection image contains "
            "no valid pixels."
        )

    return True


# ============================================================
# CONFIDENCE FILTER
# ============================================================

def filter_detections(
    detections: List[Detection],
    confidence_threshold: float,
):
    """
    Keep only detections above
    confidence threshold.
    """

    confidence_threshold = float(
        confidence_threshold
    )

    return [
        detection
        for detection in detections
        if detection.confidence
        >= confidence_threshold
    ]


# ============================================================
# CLASS FILTER
# ============================================================

def filter_classes(
    detections: List[Detection],
    selected_classes: list[str],
):
    """
    Keep only requested classes.
    """

    if not selected_classes:

        return detections

    selected = {
        item.lower()
        for item in selected_classes
    }

    return [
        detection
        for detection in detections
        if detection.label.lower()
        in selected
    ]


# ============================================================
# DETECTION SUMMARY
# ============================================================

def detection_summary(
    detections: List[Detection],
):
    """
    Generate simple class statistics.
    """

    summary = {}

    for detection in detections:

        label = detection.label

        if label not in summary:

            summary[label] = 0

        summary[label] += 1

    return summary


# ============================================================
# DRAWING
# ============================================================

def draw_detections(
    image,
    detections: List[Detection],
):
    """
    Draw detection boxes using matplotlib.

    Returns a matplotlib Figure.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle


    image = np.asarray(
        image
    )


    fig, ax = plt.subplots(
        figsize=(12, 8)
    )


    ax.imshow(
        image
    )


    for detection in detections:

        width = (
            detection.x2
            - detection.x1
        )

        height = (
            detection.y2
            - detection.y1
        )


        rectangle = Rectangle(
            (
                detection.x1,
                detection.y1,
            ),
            width,
            height,
            fill=False,
            linewidth=2,
        )


        ax.add_patch(
            rectangle
        )


        ax.text(
            detection.x1,
            max(
                detection.y1 - 5,
                5,
            ),
            (
                f"{detection.label} "
                f"{detection.confidence:.0%}"
            ),
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )


    ax.axis(
        "off"
    )


    fig.tight_layout()

    return fig