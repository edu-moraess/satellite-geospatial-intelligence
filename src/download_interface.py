"""
Interface unificada para download de bandas.
"""

from src.downloader import download_required_bands
from src.downloader_landsat import download_landsat_bands
from src.sensor_registry import get_sensor
from pathlib import Path
from typing import Optional

def download_sensor_bands(
    sensor_id: str,
    item,
    bbox: list,
    output_directory: Path
):
    """
    Baixa as bandas necessárias para o sensor especificado.
    """
    sensor = get_sensor(sensor_id)
    if sensor is None:
        raise ValueError(f"Sensor '{sensor_id}' não suportado.")

    if sensor_id == "sentinel2":
        return download_required_bands(item, bbox, output_directory)
    elif sensor_id == "landsat":
        return download_landsat_bands(item, bbox, output_directory)
    else:
        raise NotImplementedError(f"Download para '{sensor_id}' ainda não implementado.")