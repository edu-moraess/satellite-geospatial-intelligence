"""
Satellite Image Tiling
======================

Splits large satellite images into smaller
patches suitable for AI inference.
"""

from __future__ import annotations

import numpy as np


def create_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
):
    """
    Split an image into overlapping tiles.

    Returns
    -------
    list[dict]
        Each element contains:

        {
            "image": tile,
            "x": x_position,
            "y": y_position,
            "width": width,
            "height": height
        }
    """

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            "Image must have shape "
            "(height, width, channels)."
        )

    if image.shape[-1] != 3:
        raise ValueError(
            "Image must contain 3 channels."
        )

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be positive."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= tile_size:
        raise ValueError(
            "overlap must be smaller "
            "than tile_size."
        )

    height, width, _ = image.shape

    stride = tile_size - overlap

    tiles = []

    y_positions = list(
        range(
            0,
            max(height - tile_size, 0) + 1,
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            max(width - tile_size, 0) + 1,
            stride,
        )
    )

    # Guarantee coverage of image borders.
    if not y_positions or (
        y_positions[-1] + tile_size < height
    ):
        y_positions.append(
            max(height - tile_size, 0)
        )

    if not x_positions or (
        x_positions[-1] + tile_size < width
    ):
        x_positions.append(
            max(width - tile_size, 0)
        )

    for y in sorted(set(y_positions)):

        for x in sorted(set(x_positions)):

            tile = image[
                y:min(y + tile_size, height),
                x:min(x + tile_size, width),
                :,
            ]

            tiles.append(
                {
                    "image": tile,
                    "x": x,
                    "y": y,
                    "width": tile.shape[1],
                    "height": tile.shape[0],
                }
            )

    return tiles


def tile_count(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
):
    """
    Return the number of tiles that
    would be generated.
    """

    return len(
        create_tiles(
            image,
            tile_size=tile_size,
            overlap=overlap,
        )
    )