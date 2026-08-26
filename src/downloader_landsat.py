"""
Download de bandas Landsat (Collection 2 Level-2) com renomeação para compatibilidade.
"""

import rasterio
from pathlib import Path
import requests

def download_landsat_bands(item, bbox, output_dir):
    """
    Baixa as bandas Landsat e as renomeia para compatibilidade com o pipeline existente.
    Mapeamento automático dos assets disponíveis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lista de assets disponíveis na cena
    available_assets = list(item.assets.keys())

    # Mapeamento dos nomes dos assets para as bandas Sentinel-2
    # Ordem de prioridade: nomes mais comuns primeiro
    possible_names = {
        "B02": ["blue", "coastal", "aerosol", "SR_B2", "B2", "B02"],
        "B03": ["green", "SR_B3", "B3", "B03"],
        "B04": ["red", "SR_B4", "B4", "B04"],
        "B08": ["nir", "SR_B5", "B5", "B05", "B08", "near_ir"],
        "B11": ["swir1", "swir", "SR_B6", "B6", "B06", "B11"]
    }

    bands = {}
    for sentinel_key, name_options in possible_names.items():
        found = False
        for name in name_options:
            if name in item.assets:
                asset = item.assets[name]
                url = asset.href
                response = requests.get(url, stream=True)
                response.raise_for_status()

                file_path = output_dir / f"{sentinel_key}.tif"
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Validação simples com rasterio
                with rasterio.open(file_path) as src:
                    pass

                bands[sentinel_key] = file_path
                found = True
                break

        if not found:
            # Se não encontrou, levanta erro com lista de assets disponíveis
            raise ValueError(
                f"Nenhum asset encontrado para a banda {sentinel_key}.\n"
                f"Opções tentadas: {name_options}\n"
                f"Assets disponíveis na cena: {available_assets}"
            )

    return bands