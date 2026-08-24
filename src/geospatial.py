import rasterio
import numpy as np


def read_band(path):

    with rasterio.open(path) as src:

        data = src.read(1)

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "resolution": src.res,
        }

    return data, metadata


def normalize_image(image):
    image = image.astype(np.float32)

    min_value = np.nanpercentile(
        image,
        2,
    )

    max_value = np.nanpercentile(
        image,
        98,
    )

    image = (
        image - min_value
    ) / (
        max_value - min_value
    )

    return np.clip(
        image,
        0,
        1,
    )