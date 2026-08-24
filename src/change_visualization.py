"""
Change Detection Visualization
==============================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np


# ============================================================
# CHANGE MAP
# ============================================================

def create_change_figure(
    change_map,
    title="Change Detection",
):
    """
    Visualize:

    -1 = decrease
     0 = unchanged
    +1 = increase
    """

    change_map = np.asarray(
        change_map
    )

    cmap = ListedColormap(
        [
            "#D73027",  # decrease
            "#F0F0F0",  # unchanged
            "#1A9850",  # increase
        ]
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        change_map,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    axis.set_title(
        title,
        fontsize=16,
    )

    axis.set_xlabel(
        "Pixel X"
    )

    axis.set_ylabel(
        "Pixel Y"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=[
            -1,
            0,
            1,
        ],
    )

    colorbar.ax.set_yticklabels(
        [
            "Decrease",
            "No significant change",
            "Increase",
        ]
    )

    figure.tight_layout()

    return figure


# ============================================================
# CONTINUOUS DIFFERENCE
# ============================================================

def create_difference_figure(
    difference,
    title="Spectral Difference",
):
    """
    Visualize continuous difference values.
    """

    difference = np.asarray(
        difference
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        difference,
        cmap="RdBu_r",
        interpolation="nearest",
    )

    axis.set_title(
        title,
        fontsize=16,
    )

    axis.set_xlabel(
        "Pixel X"
    )

    axis.set_ylabel(
        "Pixel Y"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Difference",
    )

    figure.tight_layout()

    return figure