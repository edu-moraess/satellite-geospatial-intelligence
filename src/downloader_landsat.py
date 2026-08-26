"""
Download de bandas Landsat (Collection 2 Level-2) com renomeação para compatibilidade.
"""

import rasterio
from pathlib import Path
import requests

def download_landsat_bands(item, bbox, output_dir):
    """
    Baixa as bandas Landsat e as renomeia para compatibilidade com o pipeline existente.
    Mapeamento:
        Landsat SR_B2 → B02 (blue)
        Landsat SR_B3 → B03 (green)
        Landsat SR_B4 → B04 (red)
        Landsat SR_B5 → B08 (nir)
        Landsat SR_B6 → B11 (swir)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapeamento das bandas Landsat para os nomes esperados pelo processamento (Sentinel-2 style)
    mapping = {
        "B02": "SR_B2",  # blue
        "B03": "SR_B3",  # green
        "B04": "SR_B4",  # red
        "B08": "SR_B5",  # nir
        "B11": "SR_B6"   # swir
    }
    
    bands = {}
    for sentinel_key, landsat_key in mapping.items():
        asset = item.assets.get(landsat_key)
        if not asset:
            raise ValueError(f"Banda {landsat_key} não encontrada no item.")
        
        url = asset.href
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_path = output_dir / f"{sentinel_key}.tif"
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Validar com rasterio
        with rasterio.open(file_path) as src:
            pass
        
        bands[sentinel_key] = file_path
    
    return bands