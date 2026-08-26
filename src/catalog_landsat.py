"""
Busca no catálogo Landsat (Planetary Computer).
"""

from pystac_client import Client
import planetary_computer
from src.catalog import create_bbox  # <-- IMPORT CORRETA
from src.config import PLANETARY_COMPUTER_STAC

def search_landsat(latitude, longitude, area_size, start_date, end_date, max_cloud_cover, bbox=None):
    """Busca cenas Landsat na região e período especificados."""
    # IMPORTANTE: a assinatura das URLs (SAS token) precisa acontecer no
    # momento da conexão com o STAC, via modifier=sign_inplace — exatamente
    # como já é feito para o Sentinel-2 em src/catalog.py.
    #
    # A versão anterior chamava `sign(item)` sem capturar o valor de retorno.
    # A função `sign()` da SDK do Planetary Computer retorna o item assinado
    # em vez de garantir a mutação do objeto original in-place, então aquele
    # loop descartava o resultado e os itens continuavam com hrefs sem SAS
    # token. Isso fazia o download de qualquer banda Landsat falhar com
    # HTTPError 409 "Public access is not permitted on this storage account.".
    catalog = Client.open(
        PLANETARY_COMPUTER_STAC,
        modifier=planetary_computer.sign_inplace,
    )

    if bbox is None:
        bbox = create_bbox(latitude, longitude, area_size)

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    # Itens já saem assinados aqui, sem loop manual necessário.
    items = list(search.items())

    return items
