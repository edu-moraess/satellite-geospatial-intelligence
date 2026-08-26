"""
Registro central de sensores suportados.
Define metadados e mapeamento de bandas para cada sensor.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class SensorDefinition:
    """Definição de um sensor e suas características."""
    id: str
    name: str
    collection: str
    bands: Dict[str, str]          # mapeamento: "blue" -> "B02", etc.
    resolution: int                # resolução em metros (aproximada)
    ndvi_bands: Tuple[str, str]    # (nir, red)
    ndwi_bands: Tuple[str, str]    # (green, nir)
    ndbi_bands: Tuple[str, str]    # (swir, nir) ou similar
    supports_cloud_filter: bool = True
    description: str = ""

# Registro global de sensores
SENSORS = {
    "sentinel2": SensorDefinition(
        id="sentinel2",
        name="Sentinel-2",
        collection="sentinel-2-l2a",
        bands={
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B08",
            "swir": "B11"
        },
        resolution=10,
        ndvi_bands=("B08", "B04"),
        ndwi_bands=("B03", "B08"),
        ndbi_bands=("B11", "B08"),
        description="Copernicus Sentinel-2 Level-2A (10m, 20m, 60m)"
    ),
    "landsat": SensorDefinition(
        id="landsat",
        name="Landsat (C2 L2)",
        collection="landsat-c2-l2",
        bands={
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir": "SR_B6"
        },
        resolution=30,
        ndvi_bands=("SR_B5", "SR_B4"),
        ndwi_bands=("SR_B3", "SR_B5"),
        ndbi_bands=("SR_B6", "SR_B5"),
        description="Landsat Collection 2 Level-2 (30m, desde 1982)"
    ),
}

def get_sensor(sensor_id: str) -> Optional[SensorDefinition]:
    """Retorna a definição do sensor ou None se não existir."""
    return SENSORS.get(sensor_id)

def list_sensors() -> list[str]:
    """Retorna a lista de IDs de sensores disponíveis."""
    return list(SENSORS.keys())

def get_sensor_by_name(name: str) -> Optional[SensorDefinition]:
    """Retorna o sensor pelo nome (case-insensitive)."""
    name_lower = name.lower()
    for sensor in SENSORS.values():
        if sensor.name.lower() == name_lower:
            return sensor
    return None 