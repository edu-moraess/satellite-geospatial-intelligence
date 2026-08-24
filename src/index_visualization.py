import numpy as np
import plotly.express as px

def create_index_figure(index, title, cmap="RdYlGn"):
    display_index = np.clip(index, -1, 1)
    fig = px.imshow(display_index, zmin=-1, zmax=1, color_continuous_scale=cmap, aspect="auto")
    fig.update_layout(
        title=title, title_font=dict(color="white", size=18),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        coloraxis_colorbar=dict(title="Value", title_font_color="white", tickfont_color="white")
    )
    return fig