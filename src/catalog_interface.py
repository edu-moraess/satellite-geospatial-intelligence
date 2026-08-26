"""
Interface unificada de busca no catálogo.
Delega a busca para o módulo específico de cada sensor.
"""

from src.catalog import search_sentinel
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

    Args:
        sensor_id: Identificador do sensor (ex: "sentinel2").
        latitude, longitude: Centro da AOI.
        area_size: Tamanho da área em graus.
        start_date, end_date: Período de busca.
        max_cloud_cover: Cobertura máxima de nuvens (%).
        bbox: Opcional, bounding box (se fornecido, substitui latitude/longitude/area_size).

    Returns:
        Lista de itens (pystac.Item) ou equivalente.

    Raises:
        ValueError: Se o sensor não for suportado.
        NotImplementedError: Se a busca para o sensor não estiver implementada.
    """
    sensor = get_sensor(sensor_id)
    if sensor is None:
        raise ValueError(f"Sensor '{sensor_id}' não suportado.")

    # Delegar para o módulo correto
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
    else:
        raise NotImplementedError(f"Busca para '{sensor_id}' ainda não implementada.")