"""Landsat catalog search via Planetary Computer."""
from __future__ import annotations

import time

import planetary_computer
from pystac_client import Client

from src.catalog import create_bbox
from src.config import PLANETARY_COMPUTER_STAC


def search_landsat(
    latitude,
    longitude,
    area_size,
    start_date,
    end_date,
    max_cloud_cover,
    bbox=None,
    max_retries=3,
    max_items=30,
):
    """Search real Landsat Collection 2 Level-2 scenes with bounded retries."""
    if bbox is None:
        bbox = create_bbox(latitude, longitude, area_size)

    last_error = None
    for attempt in range(max_retries):
        try:
            catalog = Client.open(
                PLANETARY_COMPUTER_STAC,
                modifier=planetary_computer.sign_inplace,
            )
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lte": max_cloud_cover}},
                max_items=max_items,
            )
            items = list(search.items())
            items.sort(
                key=lambda item: item.properties.get("eo:cloud_cover", 100)
            )
            return items
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Landsat catalog search failed after {max_retries} attempts."
    ) from last_error
