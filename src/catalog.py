"""
Satellite catalog search.

Responsible for:
- Connecting to Planetary Computer
- Creating AOI
- Searching Sentinel-2 L2A
- Filtering by date
- Filtering by cloud coverage
"""

import planetary_computer
import pystac_client

from .config import (
    PLANETARY_COMPUTER_STAC,
    SENTINEL_COLLECTION,
)


# ============================================================
# CATALOG CONNECTION
# ============================================================

def connect_catalog():

    return pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC,
        modifier=planetary_computer.sign_inplace,
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

    return [
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ]


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
):
    """
    Search Sentinel-2 Level-2A scenes.

    If `bbox` is provided (e.g. a polygon/rectangle drawn by
    the user on the map, via src.aoi.get_selected_aoi), it is
    used directly as the search area and `latitude`/
    `longitude`/`area_size` are ignored for the bbox
    computation. Otherwise the bbox falls back to the
    lat/lon/area_size fields, exactly as before.
    """

    catalog = connect_catalog()

    if bbox is None:

        bbox = create_bbox(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
        )

    search = catalog.search(
        collections=[
            SENTINEL_COLLECTION
        ],
        bbox=bbox,
        datetime=(
            f"{start_date}/{end_date}"
        ),
        query={
            "eo:cloud_cover": {
                "lte": max_cloud_cover
            }
        },
    )

    items = list(
        search.items()
    )

    # Best cloud coverage first
    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100,
        )
    )

    return items
