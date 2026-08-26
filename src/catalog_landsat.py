"""
Busca no catálogo Landsat (Planetary Computer).
"""

from pystac_client import Client
from planetary_computer import sign
from src.catalog import create_bbox  # <-- CORRIGIDO: importa de src.catalog

def search_landsat(latitude, longitude, area_size, start_date, end_date, max_cloud_cover, bbox=None):
    """Busca cenas Landsat na região e período especificados."""
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    if bbox is None:
        bbox = create_bbox(latitude, longitude, area_size)
    
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    
    items = list(search.items())
    # Assina os itens para download
    for item in items:
        sign(item)
    
    return items