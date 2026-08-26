"""
Satellite catalog search.

Responsible for:
- Connecting to STAC catalog (AWS Earth Search)
- Creating AOI
- Searching Sentinel-2 L2A
- Filtering by date
- Filtering by cloud coverage
"""

import time
import pystac_client
from pystac_client.stac_api_io import StacApiIO

from .config import (
    PLANETARY_COMPUTER_STAC,
    SENTINEL_COLLECTION,
)


# ============================================================
# CATALOG CONNECTION
# ============================================================

def connect_catalog(timeout: int = 180):
    """
    Open STAC client with increased timeout.
    AWS endpoint não requer assinatura.
    """
    stac_io = StacApiIO(timeout=timeout)

    # Mantém compatibilidade com Planetary Computer se quiser voltar
    modifier = None
    if "planetarycomputer" in PLANETARY_COMPUTER_STAC:
        import planetary_computer
        modifier = planetary_computer.sign_inplace

    return pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC,
        modifier=modifier,
        stac_io=stac_io,
    )


# ============================================================
# CREATE BOUNDING BOX
# ============================================================

def create_bbox(
    latitude: float,
    longitude: float,
    area_size: float,
):
    """
    Create a geographic bounding box.

    area_size is expressed approximately
    in degrees.
    """
    half = area_size / 2
    min_lon = longitude - half
    min_lat = latitude - half
    max_lon = longitude + half
    max_lat = latitude + half
    return [min_lon, min_lat, max_lon, max_lat]


# ============================================================
# SEARCH SENTINEL-2
# ============================================================

def search_sentinel(
    latitude: float,
    longitude: float,
    area_size: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
    bbox=None,
    max_items: int = 30,
    max_retries: int = 3,
):
    """
    Search Sentinel-2 Level-2A scenes with retry and timeout.

    If `bbox` is provided (e.g. a polygon/rectangle drawn by
    the user on the map, via src.aoi.get_selected_aoi), it is
    used directly as the search area and `latitude`/
    `longitude`/`area_size` are ignored for the bbox
    computation. Otherwise the bbox falls back to the
    lat/lon/area_size fields.

    Additional parameters:
        max_items: limit number of results to reduce server load.
        max_retries: number of attempts before giving up.
    """
    catalog = connect_catalog(timeout=180)

    if bbox is None:
        bbox = create_bbox(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
        )

    search = catalog.search(
        collections=[SENTINEL_COLLECTION],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lte": max_cloud_cover}},
        max_items=max_items,
    )

    # Retry loop with exponential backoff
    items = None
    for attempt in range(max_retries):
        try:
            items = list(search.items())
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 2s, 4s, 8s
            print(
                f"Search failed (attempt {attempt+1}/{max_retries}): "
                f"{e}. Retrying in {wait}s..."
            )
            time.sleep(wait)

    # Best cloud coverage first
    items.sort(
        key=lambda item: item.properties.get("eo:cloud_cover", 100)
    )

    return items