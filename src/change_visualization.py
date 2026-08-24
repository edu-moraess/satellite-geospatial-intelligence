"""
Change Detection Visualization (Plotly Version)
==============================
"""

from __future__ import annotations

import numpy as np
import plotly.express as px


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

    # Cores: Vermelho (Decrease), Cinza (Unchanged), Verde (Increase)
    discrete_colorscale = [
        [0.0, "#D73027"],   # decrease
        [0.5, "#F0F0F0"],   # unchanged
        [1.0, "#1A9850"],   # increase
    ]

    fig = px.imshow(
        change_map,
        zmin=-1,
        zmax=1,
        color_continuous_scale=discrete_colorscale,
        labels=dict(x="Pixel X", y="Pixel Y", color="Change"),
        aspect="auto"
    )

    fig.update_layout(
        title=title,
        title_font=dict(color="white", size=18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        coloraxis_colorbar=dict(
            tickvals=[-1, 0, 1],
            ticktext=["Decrease", "No significant change", "Increase"],
            title="Change",
            title_font_color="white",
            tickfont_color="white",
            outlinewidth=1,
            outlinecolor="white"
        )
    )

    return fig


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

    fig = px.imshow(
        difference,
        color_continuous_scale="RdBu_r",
        labels=dict(x="Pixel X", y="Pixel Y", color="Difference"),
        aspect="auto"
    )

    fig.update_layout(
        title=title,
        title_font=dict(color="white", size=18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        coloraxis_colorbar=dict(
            title="Difference",
            title_font_color="white",
            tickfont_color="white",
            outlinewidth=1,
            outlinecolor="white"
        )
    )

    return fig