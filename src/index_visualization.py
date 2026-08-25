"""
Visualization of spectral indices.
"""

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
    Create spectral index visualization.
    """

    display_index = np.clip(
        index,
        -1,
        1,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        display_index,
        vmin=-1,
        vmax=1,
        cmap=cmap,
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Index value"
    )

    fig.tight_layout()

    return fig