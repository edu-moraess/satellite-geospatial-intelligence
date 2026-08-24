"""
Object Detection Engine
=======================

Satellite Geospatial Intelligence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:

    label: str

    confidence: float

    x1: float
    y1: float

    x2: float
    y2: float


def normalize_rgb(
    red,
    green,
    blue,
):

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


def validate_detection_image(
    image,
):

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


def filter_detections(
    detections: List[Detection],
    confidence_threshold: float,
):

    return [
        detection
        for detection in detections
        if detection.confidence
        >= confidence_threshold
    ]


def filter_classes(
    detections: List[Detection],
    selected_classes: list[str],
):

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


def detection_summary(
    detections: List[Detection],
):

    summary = {}

    for detection in detections:

        label = detection.label

        summary[label] = (
            summary.get(label, 0) + 1
        )

    return summary


def convert_model_detections(
    model_detections,
):

    result = []

    for detection in model_detections:

        result.append(
            Detection(
                label=detection.label,
                confidence=detection.confidence,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
            )
        )

    return result


def draw_detections(
    image,
    detections: List[Detection],
):

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