"""Imagery processing workflow. Scientific operations only; no Streamlit UI."""
from __future__ import annotations
import numpy as np
from src.geospatial import read_band, align_band_to_reference
from src.raster_validation import validate_raster, RasterValidationError
from src.visualization import create_rgb, create_false_color
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi
from src.index_visualization import create_index_figure
from src.classification import classify_land_cover, calculate_class_percentages
from src.land_cover import create_land_cover_figure, calculate_area_km2
from src.object_detection import normalize_rgb, validate_detection_image


def _stats(arr: np.ndarray) -> dict | None:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return {"mean": float(np.mean(finite)), "median": float(np.median(finite)), "min": float(np.min(finite)), "max": float(np.max(finite)), "std": float(np.std(finite))}


def process_scene(data: dict, resolution: float) -> dict:
    bands = data["bands"]
    b02, m02 = read_band(bands["B02"])
    b03, m03 = read_band(bands["B03"])
    b04, m04 = read_band(bands["B04"])
    b08, m08 = read_band(bands["B08"])
    b11, m11 = read_band(bands["B11"])
    b02 = align_band_to_reference(b02, m02, b04, m04)
    b03 = align_band_to_reference(b03, m03, b04, m04)
    b08 = align_band_to_reference(b08, m08, b04, m04)
    b11 = align_band_to_reference(b11, m11, b04, m04)
    for band, label in [(b02,"B02"),(b03,"B03"),(b04,"B04"),(b08,"B08"),(b11,"B11")]:
        validate_raster(band, label=label)
    ndvi = calculate_ndvi(red=b04, nir=b08)
    ndwi = calculate_ndwi(green=b03, nir=b08)
    ndbi = calculate_ndbi(nir=b08, swir=b11)
    classification = classify_land_cover(ndvi=ndvi, ndwi=ndwi, ndbi=ndbi)
    detection_rgb = normalize_rgb(red=b04, green=b03, blue=b02)
    validate_detection_image(detection_rgb)
    return {
        "rgb_img": create_rgb(blue=b02, green=b03, red=b04),
        "false_color_img": create_false_color(green=b03, red=b04, nir=b08),
        "ndvi": ndvi, "ndwi": ndwi, "ndbi": ndbi,
        "index_stats": {"ndvi": _stats(ndvi), "ndwi": _stats(ndwi), "ndbi": _stats(ndbi)},
        "index_figure": create_index_figure(ndvi, "NDVI — Vegetation", cmap="RdYlGn"),
        "classification": classification,
        "classification_fig": create_land_cover_figure(classification),
        "percentages": calculate_class_percentages(classification),
        "area_data": calculate_area_km2(classification, pixel_size_meters=float(resolution)),
        "detection_rgb": detection_rgb,
        "transform": m04["transform"], "crs": str(m04["crs"]),
    }
