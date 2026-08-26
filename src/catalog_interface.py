"""
Interface unificada de busca no catálogo.
Delega a busca para o módulo específico de cada sensor.
"""

from src.catalog import search_sentinel
from src.catalog_landsat import search_landsat
from src.sensor_registry import get_sensor
from typing import Optional

def search_sensor_catalog(
    sensor_id: str,
    latitude: float,
    longitude: float,
    area_size: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: int,
    bbox: Optional[list] = None
):
    """
    Função unificada para busca de cenas no catálogo.
    """
    sensor = get_sensor(sensor_id)
    if sensor is None:
        raise ValueError(f"Sensor '{sensor_id}' não suportado.")

    if sensor_id == "sentinel2":
        return search_sentinel(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            bbox=bbox
        )
    elif sensor_id == "landsat":
        return search_landsat(
            latitude=latitude,
            longitude=longitude,
            area_size=area_size,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            bbox=bbox
        )
    else:
        raise NotImplementedError(f"Busca para '{sensor_id}' ainda não implementada.") 