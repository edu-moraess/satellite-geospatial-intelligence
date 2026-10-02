"""
Unified satellite catalog search interface.

Delegates discovery to the implementation registered for each sensor.
"""
from typing import Optional

from src.catalog import search_sentinel
from src.catalog_landsat import search_landsat
from src.sensor_registry import get_sensor


def search_sensor_catalog(
    sensor_id: str,
    latitude: float,
    longitude: float,
    area_size: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: int,
    bbox: Optional[list] = None,
    max_retries: int = 3,
    max_items: int = 30,
):
    """Search the selected real satellite catalog with bounded retries."""
    sensor = get_sensor(sensor_id)
    if sensor is None:
        raise ValueError(f"Sensor '{sensor_id}' not supported.")

    if sensor_id == "sentinel2":
        return search_sentinel(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            bbox=bbox,
            max_retries=max_retries,
            max_items=max_items,
        )

    if sensor_id == "landsat":
        return search_landsat(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            bbox=bbox,
            max_retries=max_retries,
            max_items=max_items,
        )

    raise NotImplementedError(
        f"Search for '{sensor_id}' is not implemented."
    )
