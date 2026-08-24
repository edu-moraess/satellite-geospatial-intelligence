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
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:

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


def export_geojson(
    gdf: gpd.GeoDataFrame,
    output_path,
):

    output_path = str(
        output_path
    )

    gdf.to_file(
        output_path,
        driver="GeoJSON",
    )

    return output_path