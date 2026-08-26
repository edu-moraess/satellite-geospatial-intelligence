"""
Visualization of spectral indices.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CREATE FIGURE
# ============================================================

def create_index_figure(
    index,
    title,
    cmap="RdYlGn",
):
    """
    Create a spectral-index visualization.

    Spectral indices are clipped to the physical display range
    [-1, 1].

    Invalid pixels are rendered transparently.
    """

    index = np.asarray(
        index,
        dtype=np.float32,
    )

    display_index = np.array(
        index,
        dtype=np.float32,
        copy=True,
    )

    display_index[
        ~np.isfinite(display_index)
    ] = np.nan

    display_index = np.clip(
        display_index,
        -1.0,
        1.0,
    )

    cmap_object = plt.get_cmap(
        cmap
    ).copy()

    cmap_object.set_bad(
        alpha=0.0
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        display_index,
        vmin=-1.0,
        vmax=1.0,
        cmap=cmap_object,
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
        "Index value"
    )

    figure.tight_layout()

    return figure