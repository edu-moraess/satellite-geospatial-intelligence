from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


def download_band(
    item,
    band_name,
    bbox,
    output_directory: Path,
):
    asset = item.assets.get(band_name)

    if asset is None:
        raise ValueError(
            f"Band {band_name} not available."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{band_name}.tif"
    )

    with rasterio.open(asset.href) as src:

        raster_bbox = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox,
        )

        window = from_bounds(
            *raster_bbox,
            transform=src.transform,
        )

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        data = src.read(
            1,
            window=window,
        )

        transform = src.window_transform(
            window
        )

        profile = src.profile.copy()

        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "count": 1,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(data, 1)

    return output_path


def download_required_bands(
    item,
    bbox,
    output_directory,
):
    bands = [
        "B02",
        "B03",
        "B04",
        "B08",
    ]

    downloaded = {}

    for band in bands:

        path = download_band(
            item=item,
            band_name=band,
            bbox=bbox,
            output_directory=output_directory,
        )

        downloaded[band] = path

    return downloaded