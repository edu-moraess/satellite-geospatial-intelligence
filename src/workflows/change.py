"""Before/after change-detection workflow."""
from __future__ import annotations
import numpy as np
from src.geospatial import read_band, align_band_to_reference
from src.raster_validation import validate_raster, validate_raster_pair
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi
from src.change_detection import calculate_difference, detect_change, calculate_change_statistics
from src.change_visualization import create_change_figure


def run_change(before_bands: dict, after_bands: dict, index_name: str, threshold: float, resolution: float) -> dict:
    b04_b, m04_b = read_band(before_bands["B04"]); b03_b, m03_b = read_band(before_bands["B03"]); b08_b, m08_b = read_band(before_bands["B08"]); b11_b, m11_b = read_band(before_bands["B11"])
    b04_a, m04_a = read_band(after_bands["B04"]); b03_a, m03_a = read_band(after_bands["B03"]); b08_a, m08_a = read_band(after_bands["B08"]); b11_a, m11_a = read_band(after_bands["B11"])
    validate_raster(b04_b, m04_b, label="Before B04"); validate_raster(b04_a, m04_a, label="After B04")
    b03_b = align_band_to_reference(b03_b,m03_b,b04_b,m04_b); b08_b = align_band_to_reference(b08_b,m08_b,b04_b,m04_b); b11_b = align_band_to_reference(b11_b,m11_b,b04_b,m04_b)
    b03_a = align_band_to_reference(b03_a,m03_a,b04_a,m04_a); b08_a = align_band_to_reference(b08_a,m08_a,b04_a,m04_a); b11_a = align_band_to_reference(b11_a,m11_a,b04_a,m04_a)
    for band,label in [(b03_b,"Before B03"),(b08_b,"Before B08"),(b11_b,"Before B11"),(b03_a,"After B03"),(b08_a,"After B08"),(b11_a,"After B11")]: validate_raster(band,label=label)
    if index_name == "NDVI": before_idx = calculate_ndvi(red=b04_b,nir=b08_b); after_idx = calculate_ndvi(red=b04_a,nir=b08_a)
    elif index_name == "NDWI": before_idx = calculate_ndwi(green=b03_b,nir=b08_b); after_idx = calculate_ndwi(green=b03_a,nir=b08_a)
    else: before_idx = calculate_ndbi(nir=b08_b,swir=b11_b); after_idx = calculate_ndbi(nir=b08_a,swir=b11_a)
    validate_raster(before_idx,label=f"Before {index_name}"); validate_raster(after_idx,label=f"After {index_name}")
    after_aligned = align_band_to_reference(after_idx,m04_a,before_idx,m04_b)
    before_meta = m04_b.copy(); before_meta.update({"height":before_idx.shape[0],"width":before_idx.shape[1],"transform":m04_b["transform"],"crs":m04_b["crs"],"nodata":np.nan,"dtype":str(before_idx.dtype)})
    after_meta = m04_b.copy(); after_meta.update({"height":after_aligned.shape[0],"width":after_aligned.shape[1],"transform":m04_b["transform"],"crs":m04_b["crs"],"nodata":np.nan,"dtype":str(after_aligned.dtype)})
    validation = validate_raster_pair(before_idx,after_aligned,metadata_a=before_meta,metadata_b=after_meta,label_a=f"Before {index_name}",label_b=f"After {index_name}",require_same_dtype=False)
    diff = calculate_difference(before=before_idx,after=after_aligned,before_metadata=before_meta,after_metadata=after_meta); validate_raster(diff,label=f"{index_name} difference")
    change_map = detect_change(diff,threshold=threshold)
    stats = calculate_change_statistics(change_map,pixel_size_meters=float(resolution))
    return {"statistics":stats,"figure":create_change_figure(change_map,title=f"{index_name} Change Detection"),"validation":validation,"alignment":{"reference":"Before","target":"After","shape":before_idx.shape,"overlap_fraction":validation["overlap_fraction"]}}
