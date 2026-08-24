"""
Visualization of spectral indices (Plotly Version).
"""

import numpy as np
import plotly.express as px


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

    fig = px.imshow(
        display_index,
        zmin=-1,
        zmax=1,
        color_continuous_scale=cmap,
        labels=dict(x="", y="", color="Index value"),
        aspect="auto"
    )

    # Ajuste para visual Dark Mode Profissional
    fig.update_layout(
        title=title,
        title_font=dict(color="white", size=18),
        plot_bgcolor="rgba(0,0,0,0)",   # Fundo do gráfico transparente
        paper_bgcolor="rgba(0,0,0,0)",  # Fundo da "folha" transparente
        font=dict(color="white"),
        # Remover ticks e margens para imagem ficar limpa
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        coloraxis_colorbar=dict(
            title="Index value",
            title_font_color="white",
            tickfont_color="white",
            outlinewidth=1,
            outlinecolor="white"
        )
    )

    return fig