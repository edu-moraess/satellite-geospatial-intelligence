"""Imagery processing workflow. Scientific operations only; no Streamlit UI."""
from __future__ import annotations

import numpy as np
from rasterio.enums import Resampling

from src.geospatial import read_band, align_band_to_reference, align_array_with_metadata
from src.raster_validation import validate_raster
from src.visualization import create_rgb, create_false_color
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi
from src.index_visualization import create_index_figure
from src.classification import classify_land_cover, calculate_class_percentages
from src.land_cover import create_land_cover_figure, calculate_area_km2
from src.object_detection import normalize_rgb, validate_detection_image
from src.scene_quality import scene_quality_report


def _stats(arr: np.ndarray) -> dict | None:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
        "valid_fraction": float(finite.size / arr.size),
    }


def process_scene(data: dict, resolution: float) -> dict:
    bands = data["bands"]

    b02, m02 = read_band(bands["B02"])
    b03, m03 = read_band(bands["B03"])
    b04, m04 = read_band(bands["B04"])
    b08, m08 = read_band(bands["B08"])
    b8a, m8a = read_band(bands["B8A"])
    b11, m11 = read_band(bands["B11"])
    b12, m12 = read_band(bands["B12"])
    scl, mscl = read_band(bands["SCL"])

    b02 = align_band_to_reference(b02, m02, b04, m04)
    b03 = align_band_to_reference(b03, m03, b04, m04)
    b08 = align_band_to_reference(b08, m08, b04, m04)
    b8a = align_band_to_reference(b8a, m8a, b04, m04)
    b11 = align_band_to_reference(b11, m11, b04, m04)
    b12 = align_band_to_reference(b12, m12, b04, m04)
    scl = align_array_with_metadata(
        scl,
        mscl,
        b04,
        m04,
        resampling=Resampling.nearest,
    )

    for band, label in [
        (b02, "B02"),
        (b03, "B03"),
        (b04, "B04"),
        (b08, "B08"),
        (b8a, "B8A"),
        (b11, "B11"),
        (b12, "B12"),
    ]:
        validate_raster(band, label=label)

    quality = scene_quality_report(scl, b04)

    # Mask cloud, cloud-shadow, snow and invalid SCL classes before
    # quantitative spectral analysis. RGB rendering keeps the same mask.
    invalid = ~np.isin(scl.astype(np.int16), [4, 5, 6, 7])
    b02 = b02.copy()
    b03 = b03.copy()
    b04 = b04.copy()
    b08 = b08.copy()
    b8a = b8a.copy()
    b11 = b11.copy()
    b12 = b12.copy()

    for array in (b02, b03, b04, b08, b8a, b11, b12):
        array[invalid] = np.nan

    ndvi = calculate_ndvi(red=b04, nir=b08)
    ndwi = calculate_ndwi(green=b03, nir=b08)
    ndbi = calculate_ndbi(nir=b08, swir=b11)

    classification = classify_land_cover(
        ndvi=ndvi,
        ndwi=ndwi,
        ndbi=ndbi,
    )

    detection_rgb = normalize_rgb(
        red=b04,
        green=b03,
        blue=b02,
    )
    validate_detection_image(detection_rgb)

    return {
        "rgb_img": create_rgb(blue=b02, green=b03, red=b04),
        "false_color_img": create_false_color(green=b03, red=b04, nir=b08),
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi": ndbi,
        "index_stats": {
            "ndvi": _stats(ndvi),
            "ndwi": _stats(ndwi),
            "ndbi": _stats(ndbi),
        },
        "band_stats": {
            "B02": _stats(b02),
            "B03": _stats(b03),
            "B04": _stats(b04),
            "B08": _stats(b08),
            "B8A": _stats(b8a),
            "B11": _stats(b11),
            "B12": _stats(b12),
        },
        "scene_quality": quality,
        "scl": scl,
        "index_figure": create_index_figure(
            ndvi,
            "NDVI — Vegetation",
            cmap="RdYlGn",
        ),
        "classification": classification,
        "classification_fig": create_land_cover_figure(classification),
        "percentages": calculate_class_percentages(classification),
        "area_data": calculate_area_km2(
            classification,
            pixel_size_meters=float(resolution),
        ),
        "detection_rgb": detection_rgb,
        "transform": m04["transform"],
        "crs": str(m04["crs"]),
    }
