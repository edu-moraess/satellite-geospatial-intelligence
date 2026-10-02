from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

HLS_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
HLS_COLLECTION = "hls2-s30"
BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")


@dataclass(frozen=True)
class HLSScene:
    item_id: str
    datetime: object
    cube: np.ndarray
    transform: object
    crs: object


def search_hls_s30(*, bbox: list[float], target_datetime, max_items: int = 10):
    catalog = pystac_client.Client.open(
        HLS_STAC,
        modifier=planetary_computer.sign_inplace,
    )
    start = target_datetime - timedelta(days=1)
    end = target_datetime + timedelta(days=1)
    search = catalog.search(
        collections=[HLS_COLLECTION],
        bbox=bbox,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
    )
    items = list(search.items())
    items.sort(
        key=lambda item: abs(
            (item.datetime.replace(tzinfo=None) - target_datetime.replace(tzinfo=None)).total_seconds()
        )
    )
    return items


def load_hls_s30_item(item, bbox: list[float]) -> HLSScene:
    arrays = []
    reference_transform = None
    reference_crs = None

    for band in BANDS:
        asset = item.assets.get(band)
        if asset is None:
            raise ValueError(f"HLS S30 item {item.id} is missing {band}.")

        with rasterio.open(asset.href) as src:
            if reference_transform is None:
                reference_crs = src.crs
                rb = transform_bounds("EPSG:4326", src.crs, *bbox)
                left = max(rb[0], src.bounds.left)
                bottom = max(rb[1], src.bounds.bottom)
                right = min(rb[2], src.bounds.right)
                top = min(rb[3], src.bounds.top)
                if left >= right or bottom >= top:
                    raise ValueError("AOI does not overlap HLS S30.")
                window = from_bounds(left, bottom, right, top, transform=src.transform)
                window = window.round_offsets().round_lengths()
                reference_transform = src.window_transform(window)

            data = src.read(1, window=window).astype(np.float32)
            if src.nodata is not None:
                data[data == float(src.nodata)] = np.nan
            arrays.append(data * 0.0001)

    return HLSScene(
        item_id=item.id,
        datetime=item.datetime,
        cube=np.stack(arrays, axis=0),
        transform=reference_transform,
        crs=reference_crs,
    )
