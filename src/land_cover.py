import numpy as np
import plotly.express as px

def create_land_cover_figure(classification):
    classification = np.asarray(classification)
    discrete_colorscale = [
        [0.0, "#BDBDBD"], [0.25, "#2E7D32"], [0.50, "#1976D2"], [0.75, "#D84315"], [1.00, "#C49A6C"]
    ]
    fig = px.imshow(classification, color_continuous_scale=discrete_colorscale, zmin=0, zmax=4, aspect="auto")
    fig.update_layout(
        title="Land Cover Classification", title_font=dict(color="white", size=18),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        coloraxis_colorbar=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Other", "Vegetation", "Water", "Built-up", "Bare Soil"],
            title="Classes", title_font_color="white", tickfont_color="white"
        )
    )
    return fig

def calculate_area_km2(classification, pixel_size_meters=10.0):
    classification = np.asarray(classification)
    pixel_area_km2 = (pixel_size_meters ** 2) / 1_000_000.0
    classes = {0: "Other", 1: "Vegetation", 2: "Water", 3: "Built-up", 4: "Bare Soil"}
    result = {}
    for class_id, name in classes.items():
        pixels = np.sum(classification == class_id)
        result[name] = float(pixels) * pixel_area_km2
    return result