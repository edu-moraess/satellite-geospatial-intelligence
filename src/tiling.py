from __future__ import annotations

import numpy as np


def _validate_parameters(
    tile_size: int,
    overlap: int,
):

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= tile_size:
        raise ValueError(
            "overlap must be smaller than tile_size."
        )


def create_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[dict]:

    _validate_parameters(
        tile_size,
        overlap,
    )

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    image = np.asarray(image)

    if image.ndim not in (2, 3):

        raise ValueError(
            "Image must have 2 or 3 dimensions."
        )

    height = image.shape[0]
    width = image.shape[1]

    stride = tile_size - overlap

    tiles = []

    y_positions = list(
        range(
            0,
            max(height - tile_size + 1, 1),
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            max(width - tile_size + 1, 1),
            stride,
        )
    )

    # Guarantee coverage of image borders.

    if height > tile_size:

        last_y = height - tile_size

        if last_y not in y_positions:
            y_positions.append(last_y)

    if width > tile_size:

        last_x = width - tile_size

        if last_x not in x_positions:
            x_positions.append(last_x)


    for y in y_positions:

        for x in x_positions:

            tile = image[
                y:y + tile_size,
                x:x + tile_size,
            ]

            actual_height = tile.shape[0]
            actual_width = tile.shape[1]

            # ------------------------------------------------
            # Pad incomplete border tiles.
            # ------------------------------------------------

            if (
                actual_height < tile_size
                or actual_width < tile_size
            ):

                if image.ndim == 3:

                    padded = np.zeros(
                        (
                            tile_size,
                            tile_size,
                            image.shape[2],
                        ),
                        dtype=image.dtype,
                    )

                    padded[
                        :actual_height,
                        :actual_width,
                        :
                    ] = tile

                else:

                    padded = np.zeros(
                        (
                            tile_size,
                            tile_size,
                        ),
                        dtype=image.dtype,
                    )

                    padded[
                        :actual_height,
                        :actual_width,
                    ] = tile

                tile = padded


            tiles.append(
                {
                    "image": tile,
                    "x": x,
                    "y": y,
                    "width": actual_width,
                    "height": actual_height,
                    "tile_size": tile_size,
                }
            )

    return tiles


def tile_count(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> int:

    return len(
        create_tiles(
            image=image,
            tile_size=tile_size,
            overlap=overlap,
        )
    )