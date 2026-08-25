from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import Affine
from rasterio.warp import transform_bounds


@dataclass
class GeoDetection:
    label: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    geometry: object | None = None

    longitude: float | None = None
    latitude: float | None = None


def pixel_to_map(
    transform: Affine,
    x: float,
    y: float,
):
    """
    Converts raster pixel coordinates
    into map coordinates.
    """

    longitude, latitude = transform * (
        x,
        y,
    )

    return longitude, latitude


def detection_to_geometry(
    detection,
    transform: Affine,
    crs: str = "EPSG:4326",
):
    """
    Converts a pixel bounding box into
    a georeferenced polygon.
    """

    x1 = float(detection["x1"])
    y1 = float(detection["y1"])
    x2 = float(detection["x2"])
    y2 = float(detection["y2"])

    p1 = transform * (x1, y1)
    p2 = transform * (x2, y2)

    min_x = min(p1[0], p2[0])
    max_x = max(p1[0], p2[0])

    min_y = min(p1[1], p2[1])
    max_y = max(p1[1], p2[1])

    geometry = box(
        min_x,
        min_y,
        max_x,
        max_y,
    )

    return geometry


def georeference_detections(
    detections: Iterable,
    transform: Affine,
    crs: str = "EPSG:32723",
) -> gpd.GeoDataFrame:
    """
    Convert pixel-space detections into a GeoDataFrame.

    `crs` must be the ACTUAL CRS the `transform` is expressed
    in - normally the Sentinel-2 scene's native UTM CRS (e.g.
    "EPSG:32723"), taken from the band metadata returned by
    geospatial.read_band (metadata["crs"]). Passing a wrong
    CRS here does not raise an error, but silently mislabels
    the coordinates - the values would still be UTM meters
    while claiming to be a different CRS.

    For GeoJSON export, use `to_wgs84()` afterwards - GeoJSON
    is conventionally WGS84 (EPSG:4326) longitude/latitude,
    not the scene's native UTM meters.
    """

    records = []

    for detection in detections:

        geometry = detection_to_geometry(
            detection,
            transform,
            crs=crs,
        )

        centroid = geometry.centroid

        records.append(
            {
                "label": detection.get(
                    "label",
                    "unknown",
                ),

                "confidence": float(
                    detection.get(
                        "confidence",
                        0.0,
                    )
                ),

                "x1": float(
                    detection["x1"]
                ),

                "y1": float(
                    detection["y1"]
                ),

                "x2": float(
                    detection["x2"]
                ),

                "y2": float(
                    detection["y2"]
                ),

                "longitude": centroid.x,

                "latitude": centroid.y,

                "geometry": geometry,
            }
        )

    if not records:

        return gpd.GeoDataFrame(
            columns=[
                "label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "longitude",
                "latitude",
                "geometry",
            ],
            geometry="geometry",
            crs=crs,
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )


def detection_summary_geo(
    gdf: gpd.GeoDataFrame,
) -> dict:

    if gdf.empty:

        return {
            "objects": 0,
            "classes": 0,
        }

    return {
        "objects": len(gdf),
        "classes": int(
            gdf["label"].nunique()
        ),
    }


def to_wgs84(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Reproject a detections GeoDataFrame to WGS84
    (longitude/latitude), which is what the GeoJSON spec
    expects. Also refreshes the longitude/latitude columns
    from the reprojected centroid.
    """

    if gdf.empty or gdf.crs is None:
        return gdf

    reprojected = gdf.to_crs("EPSG:4326")

    reprojected["longitude"] = reprojected.geometry.centroid.x
    reprojected["latitude"] = reprojected.geometry.centroid.y

    return reprojected


def to_geojson_bytes(
    gdf: gpd.GeoDataFrame,
) -> bytes:
    """
    Serialize a detections GeoDataFrame to GeoJSON bytes,
    reprojecting to WGS84 first if needed. Meant for
    st.download_button, without writing to disk.
    """

    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = to_wgs84(gdf)

    return gdf.to_json().encode("utf-8")


def export_geojson(
    gdf: gpd.GeoDataFrame,
    output_path,
):
    """
    Write a detections GeoDataFrame to a .geojson file on
    disk, reprojecting to WGS84 first if needed.
    """

    output_path = str(
        output_path
    )

    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = to_wgs84(gdf)

    gdf.to_file(
        output_path,
        driver="GeoJSON",
    )

    return output_path