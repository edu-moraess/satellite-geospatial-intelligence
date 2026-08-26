"""
Download de bandas Landsat (Collection 2 Level-2) com renomeação para
compatibilidade com o pipeline existente (mesmos nomes de banda do
Sentinel-2: B02, B03, B04, B08, B11).

IMPORTANTE: assim como o downloader do Sentinel-2 (src/downloader.py), este
módulo baixa APENAS a janela (window) correspondente à AOI, lendo o COG
remoto via HTTP range requests (rasterio/GDAL) — em vez de baixar a cena
inteira.

Isso corrige o problema de download extremamente lento: uma cena Landsat
cobre ~180km x ~180km a 30m de resolução, então baixar a banda inteira via
`requests.get` (como a versão anterior fazia) significa transferir dezenas
de MB por banda mesmo quando a AOI selecionada tem poucos km de lado.
"""

from pathlib import Path
import time

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from src.downloader import ensure_output_directory, is_valid_geotiff


# ============================================================
# CONSTANTS
# ============================================================

MAX_RETRIES = 3

RETRY_DELAY_SECONDS = 2

# Mapeamento dos nomes de asset Landsat para as bandas "estilo Sentinel-2"
# usadas pelo restante do pipeline.
POSSIBLE_NAMES = {
    "B02": ["blue", "coastal", "aerosol", "SR_B2", "B2", "B02"],
    "B03": ["green", "SR_B3", "B3", "B03"],
    "B04": ["red", "SR_B4", "B4", "B04"],
    "B08": ["nir", "SR_B5", "B5", "B05", "B08", "near_ir"],
    "B11": ["swir1", "swir", "SR_B6", "B6", "B06", "B11"],
}


# ============================================================
# FIND ASSET
# ============================================================

def _find_asset(item, sentinel_key):
    """Localiza o asset Landsat correspondente a uma banda 'estilo Sentinel-2'."""

    for name in POSSIBLE_NAMES[sentinel_key]:

        if name in item.assets:

            return item.assets[name]

    return None


# ============================================================
# DOWNLOAD SINGLE BAND (WINDOWED)
# ============================================================

def _download_windowed_band(item, sentinel_key, asset, bbox, output_directory):
    """
    Baixa apenas a janela do AOI de uma banda Landsat (COG remoto).

    Mesma estratégia usada para o Sentinel-2 em src/downloader.py
    (download_band): abre o raster remoto, recorta pela janela da AOI e
    grava só esse recorte localmente.
    """

    output_path = output_directory / f"{sentinel_key}.tif"

    # --------------------------------------------------------
    # REUSE
    # --------------------------------------------------------

    if is_valid_geotiff(output_path):
        return output_path

    if output_path.exists():
        output_path.unlink(missing_ok=True)

    href = asset.href

    if not href:
        raise ValueError(f"Asset URL unavailable for {sentinel_key}.")

    # --------------------------------------------------------
    # OPEN REMOTE DATASET
    # --------------------------------------------------------

    try:

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            GDAL_HTTP_MAX_RETRY=3,
            GDAL_HTTP_RETRY_DELAY=1,
        ):

            with rasterio.open(href) as src:

                raster_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)

                left = max(raster_bbox[0], src.bounds.left)
                bottom = max(raster_bbox[1], src.bounds.bottom)
                right = min(raster_bbox[2], src.bounds.right)
                top = min(raster_bbox[3], src.bounds.top)

                if left >= right or bottom >= top:
                    raise ValueError(f"AOI does not overlap band {sentinel_key}.")

                window = from_bounds(left, bottom, right, top, transform=src.transform)
                window = window.round_offsets().round_lengths()

                if window.width <= 0 or window.height <= 0:
                    raise ValueError(f"Invalid raster window for {sentinel_key}.")

                data = None
                last_error = None

                for attempt in range(1, MAX_RETRIES + 1):

                    try:
                        data = src.read(1, window=window)
                        break

                    except Exception as error:
                        last_error = error

                        if attempt < MAX_RETRIES:
                            time.sleep(RETRY_DELAY_SECONDS * attempt)

                if data is None:
                    raise RuntimeError(
                        f"Failed to read {sentinel_key} after "
                        f"{MAX_RETRIES} attempts.\n\n"
                        f"Last error: {last_error}"
                    )

                transform = src.window_transform(window)

                profile = src.profile.copy()
                profile.update(
                    {
                        "driver": "GTiff",
                        "height": data.shape[0],
                        "width": data.shape[1],
                        "transform": transform,
                        "count": 1,
                        "compress": "deflate",
                        "dtype": str(data.dtype),
                    }
                )

    except Exception as error:
        raise RuntimeError(
            f"Failed downloading {sentinel_key}.\n\n"
            f"Scene: {item.id}\n"
            f"Asset: {href}\n\n"
            f"Error: {error}"
        ) from error

    # --------------------------------------------------------
    # WRITE LOCAL FILE
    # --------------------------------------------------------

    try:

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    except Exception as error:
        output_path.unlink(missing_ok=True)

        raise RuntimeError(
            f"Failed writing {output_path}.\n\nError: {error}"
        ) from error

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if not is_valid_geotiff(output_path):
        output_path.unlink(missing_ok=True)

        raise RuntimeError(f"Downloaded file is invalid: {output_path}")

    return output_path


# ============================================================
# DOWNLOAD REQUIRED BANDS
# ============================================================

def download_landsat_bands(item, bbox, output_dir):
    """
    Baixa apenas a janela do AOI das bandas Landsat necessárias, já
    renomeadas para compatibilidade com o pipeline (B02, B03, B04, B08, B11).
    """

    output_directory = ensure_output_directory(Path(output_dir))

    available_assets = list(item.assets.keys())

    bands = {}

    for sentinel_key in POSSIBLE_NAMES:

        asset = _find_asset(item, sentinel_key)

        if asset is None:
            raise ValueError(
                f"Nenhum asset encontrado para a banda {sentinel_key}.\n"
                f"Opções tentadas: {POSSIBLE_NAMES[sentinel_key]}\n"
                f"Assets disponíveis na cena: {available_assets}"
            )

        bands[sentinel_key] = _download_windowed_band(
            item=item,
            sentinel_key=sentinel_key,
            asset=asset,
            bbox=bbox,
            output_directory=output_directory,
        )

    return bands
