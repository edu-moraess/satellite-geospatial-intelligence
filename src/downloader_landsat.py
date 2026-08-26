"""
Download de bandas Landsat (Collection 2 Level-2) com renomeação para compatibilidade.
"""

import rasterio
from pathlib import Path
import requests
import streamlit as st

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
    
    # Mapeamento das bandas Landsat para os nomes esperados pelo processamento
    # Lista de possíveis nomes de assets para cada banda
    possible_names = {
        "B02": ["SR_B2", "B2", "B02", "blue"],          # azul
        "B03": ["SR_B3", "B3", "B03", "green"],         # verde
        "B04": ["SR_B4", "B4", "B04", "red"],           # vermelho
        "B08": ["SR_B5", "B5", "B05", "nir"],           # infravermelho próximo
        "B11": ["SR_B6", "B6", "B06", "swir1", "swir"]  # infravermelho de ondas curtas
    }
    
    # Obter lista de assets disponíveis para depuração
    available_assets = list(item.assets.keys())
    st.info(f"Assets disponíveis na cena: {available_assets}")
    
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
                
                # Validar com rasterio
                with rasterio.open(file_path) as src:
                    pass
                
                bands[sentinel_key] = file_path
                found = True
                break
        
        if not found:
            raise ValueError(
                f"Nenhum asset encontrado para a banda {sentinel_key}. "
                f"Opções tentadas: {name_options}. "
                f"Assets disponíveis: {available_assets}"
            )
    
    return bands