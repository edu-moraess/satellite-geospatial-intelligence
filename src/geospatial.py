Geospatial utilities for Satellite Geospatial Intelligence.

"""
Provides raster reading, alignment, resampling,
normalization and band stacking utilities.
"""

from future import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.raster_validation import (
RasterValidationError,
validate_raster,
)

def read_band(path):
"""
Read a single-band GeoTIFF.

Returns:
    tuple:
        array,
        metadata
"""

path = Path(path)

if not path.exists():
    raise FileNotFoundError(
        f"Band file not found: {path}"
    )

if not path.is_file():
    raise FileNotFoundError(
        f"Band path is not a file: {path}"
    )

try:
    with rasterio.open(path) as src:
        data = src.read(1).astype(
            np.float32,
            copy=False,
        )

        metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": src.bounds,
            "profile": src.profile.copy(),
        }

except Exception as error:
    raise RuntimeError(
        f"Failed to read raster band: {path}\n\n"
        f"Error: {error}"
    ) from error

validate_raster(
    data,
    metadata,
    label=path.name,
)

return data, metadata

def resample_to_reference(
source_array,
source_metadata,
reference_metadata,
resampling=Resampling.bilinear,
):
"""
Resample a raster onto the exact grid of a reference raster.

The output has the reference raster's:
    - width;
    - height;
    - CRS;
    - transform.
"""

validate_raster(
    source_array,
    source_metadata,
    label="source raster",
)

if source_metadata is None:
    raise RasterValidationError(
        "source raster: metadata is required for resampling."
    )

if reference_metadata is None:
    raise RasterValidationError(
        "reference raster: metadata is required for resampling."
    )

required_source = (
    "transform",
    "crs",
)

required_reference = (
    "transform",
    "crs",
    "width",
    "height",
)

for key in required_source:
    if source_metadata.get(key) is None:
        raise RasterValidationError(
            f"source raster metadata is missing '{key}'."
        )

for key in required_reference:
    if reference_metadata.get(key) is None:
        raise RasterValidationError(
            f"reference raster metadata is missing '{key}'."
        )

destination = np.full(
    (
        int(reference_metadata["height"]),
        int(reference_metadata["width"]),
    ),
    np.nan,
    dtype=np.float32,
)

source_array = np.asarray(
    source_array,
    dtype=np.float32,
)

source_nodata = source_metadata.get(
    "nodata"
)

try:
    reproject(
        source=source_array,
        destination=destination,
        src_transform=source_metadata["transform"],
        src_crs=source_metadata["crs"],
        dst_transform=reference_metadata["transform"],
        dst_crs=reference_metadata["crs"],
        src_nodata=source_nodata,
        dst_nodata=np.nan,
        resampling=resampling,
    )

except Exception as error:
    raise RasterValidationError(
        "Raster resampling failed.\n\n"
        f"Source CRS: {source_metadata.get('crs')}\n"
        f"Target CRS: {reference_metadata.get('crs')}\n"
        f"Error: {error}"
    ) from error

return destination

def align_band_to_reference(
band_array,
band_metadata,
reference_array,
reference_metadata,
):
"""
Align a raster to a reference raster.

If both already share the same shape, CRS and transform,
the original array is returned unchanged.

Otherwise the band is reprojected/resampled to the
reference grid.
"""

validate_raster(
    band_array,
    band_metadata,
    label="band",
)

validate_raster(
    reference_array,
    reference_metadata,
    label="reference raster",
)

same_shape = (
    np.asarray(band_array).shape
    == np.asarray(reference_array).shape
)

same_transform = (
    band_metadata.get("transform")
    == reference_metadata.get("transform")
)

same_crs = (
    band_metadata.get("crs")
    == reference_metadata.get("crs")
)

if (
    same_shape
    and same_transform
    and same_crs
):
    return np.asarray(
        band_array,
        dtype=np.float32,
    )

return resample_to_reference(
    source_array=band_array,
    source_metadata=band_metadata,
    reference_metadata=reference_metadata,
    resampling=Resampling.bilinear,
)

def align_array_with_metadata(
array,
metadata,
reference_array,
reference_metadata,
resampling=Resampling.bilinear,
):
"""
Compatibility wrapper for application code.

Equivalent to align_band_to_reference(), but explicitly
accepts a resampling method.
"""

validate_raster(
    array,
    metadata,
    label="array",
)

validate_raster(
    reference_array,
    reference_metadata,
    label="reference array",
)

same_shape = (
    np.asarray(array).shape
    == np.asarray(reference_array).shape
)

same_transform = (
    metadata.get("transform")
    == reference_metadata.get("transform")
)

same_crs = (
    metadata.get("crs")
    == reference_metadata.get("crs")
)

if (
    same_shape
    and same_transform
    and same_crs
):
    return np.asarray(
        array,
        dtype=np.float32,
    )

return resample_to_reference(
    source_array=array,
    source_metadata=metadata,
    reference_metadata=reference_metadata,
    resampling=resampling,
)

def normalize_image(image):
"""
Normalize an image to the 0-1 range using
robust 2nd/98th percentiles.
"""

image = np.asarray(
    image,
    dtype=np.float32,
)

if image.size == 0:
    return np.zeros_like(
        image,
        dtype=np.float32,
    )

valid = image[
    np.isfinite(image)
]

if valid.size == 0:
    return np.zeros_like(
        image,
        dtype=np.float32,
    )

low = float(
    np.percentile(
        valid,
        2,
    )
)

high = float(
    np.percentile(
        valid,
        98,
    )
)

if high <= low:
    result = np.zeros_like(
        image,
        dtype=np.float32,
    )

    finite = np.isfinite(image)

    result[finite] = 0.5

    return result

normalized = (
    image - low
) / (
    high - low
)

normalized = np.clip(
    normalized,
    0.0,
    1.0,
)

normalized[
    ~np.isfinite(image)
] = 0.0

return normalized.astype(
    np.float32,
    copy=False,
)

def stack_bands(bands):
"""
Stack aligned bands into a H x W x N array.

All bands must already share identical dimensions.
"""

if bands is None or len(bands) == 0:
    raise ValueError(
        "No bands provided."
    )

arrays = [
    np.asarray(
        band,
        dtype=np.float32,
    )
    for band in bands
]

shapes = {
    array.shape
    for array in arrays
}

if len(shapes) != 1:
    raise RasterValidationError(
        "All bands must have identical spatial "
        "dimensions before stacking.\n\n"
        f"Shapes found: {sorted(shapes)}"
    )

for index, array in enumerate(arrays):
    validate_raster(
        array,
        label=f"band {index + 1}",
    )

return np.stack(
    arrays,
    axis=-1,
)