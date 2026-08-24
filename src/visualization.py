import numpy as np
import matplotlib.pyplot as plt

from .geospatial import (
    read_band,
    normalize_image,
)


def create_rgb(
    blue_path,
    green_path,
    red_path,
):
    blue, _ = read_band(blue_path)
    green, _ = read_band(green_path)
    red, _ = read_band(red_path)

    blue = normalize_image(blue)
    green = normalize_image(green)
    red = normalize_image(red)

    rgb = np.dstack(
        [
            red,
            green,
            blue,
        ]
    )

    return rgb


def create_false_color(
    green_path,
    red_path,
    nir_path,
):
    green, _ = read_band(green_path)
    red, _ = read_band(red_path)
    nir, _ = read_band(nir_path)

    green = normalize_image(green)
    red = normalize_image(red)
    nir = normalize_image(nir)

    false_color = np.dstack(
        [
            nir,
            red,
            green,
        ]
    )

    return false_color


def save_image(
    image,
    output_path,
    title,
):
    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()