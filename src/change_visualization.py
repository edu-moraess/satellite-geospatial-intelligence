"""
Change Detection Visualization
==============================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
        NaN = invalid / unavailable
    """

    change_map = np.asarray(
        change_map,
        dtype=np.float32,
    )

    display_map = np.array(
        change_map,
        dtype=np.float32,
        copy=True,
    )

    display_map[
        ~np.isfinite(display_map)
    ] = np.nan

    cmap = ListedColormap(
        [
            "#D73027",
            "#F0F0F0",
            "#1A9850",
        ]
    )

    cmap.set_bad(
        alpha=0.0
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        display_map,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    axis.set_title(
        title,
        fontsize=15,
        fontweight="bold",
    )

    axis.axis(
        "off"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=[
            -1,
            0,
            1,
        ],
        fraction=0.046,
        pad=0.04,
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
    Visualize continuous spectral difference.

    NaN pixels are rendered transparently.
    """

    difference = np.asarray(
        difference,
        dtype=np.float32,
    )

    display_difference = np.array(
        difference,
        dtype=np.float32,
        copy=True,
    )

    display_difference[
        ~np.isfinite(
            display_difference
        )
    ] = np.nan

    cmap = plt.get_cmap(
        "RdBu_r"
    ).copy()

    cmap.set_bad(
        alpha=0.0
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        display_difference,
        cmap=cmap,
        interpolation="nearest",
    )

    axis.set_title(
        title,
        fontsize=15,
        fontweight="bold",
    )

    axis.axis(
        "off"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Difference"
    )

    figure.tight_layout()

    return figure