import pystac_client
import planetary_computer


from .config import PLANETARY_COMPUTER_STAC


def connect_catalog():
    return pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC,
        modifier=planetary_computer.sign_inplace,
    )


def create_bbox(
    latitude: float,
    longitude: float,
    area_size: float,
):
    half = area_size / 2

    return [
        longitude - half,
        latitude - half,
        longitude + half,
        latitude + half,
    ]


def search_sentinel(
    latitude: float,
    longitude: float,
    area_size: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
):
    catalog = connect_catalog()

    bbox = create_bbox(
        latitude,
        longitude,
        area_size,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lte": max_cloud_cover
            }
        },
    )

    items = list(search.items())

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100
        )
    )

    return items