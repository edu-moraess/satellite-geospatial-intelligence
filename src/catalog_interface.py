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
    bbox: Optional[list] = None,
    max_retries: int = 3,
    max_items: int = 30,
):
    """
    Função unificada para busca de cenas no catálogo.

    Parâmetros adicionais:
        max_retries: número de tentativas em caso de timeout/erro.
        max_items: limite de resultados retornados (reduz carga no servidor).
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
            bbox=bbox,
            max_retries=max_retries,
            max_items=max_items,
        )
    elif sensor_id == "landsat":
        # Se o módulo Landsat não aceitar os parâmetros, ajuste aqui ou remova-os
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
    else:
        raise NotImplementedError(
            f"Busca para '{sensor_id}' ainda não implementada."
        )