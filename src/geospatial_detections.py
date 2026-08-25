"""
Georeferenced export of AI object detections.

Turns pixel-space bounding boxes (from the detector)
into a GeoDataFrame / GeoJSON that can be downloaded
and opened in QGIS, ArcGIS, etc.
"""

from __future__ import annotations

from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import box


def detections_to_geodataframe(
    detections: Iterable[dict],
    transform,
    crs,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Convert a list of detection dicts (with keys
    x_min, y_min, x_max, y_max, label, score) into a
    GeoDataFrame in the raster CRS.

    `transform` is an affine transform (rasterio or
    affine.Affine) mapping pixel coordinates to the
    CRS of the source image.
    """

    detections = list(detections)

    if not detections:
        return gpd.GeoDataFrame(
            columns=["label", "score", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    records = []

    for det in detections:
        x_min = float(det["x_min"])
        y_min = float(det["y_min"])
        x_max = float(det["x_max"])
        y_max = float(det["y_max"])

        # rasterio/affine: (col, row) -> (x, y)
        # box corners in pixel space
        xs = [x_min, x_max, x_max, x_min]
        ys = [y_min, y_min, y_max, y_max]

        world_xs = []
        world_ys = []

        for col, row in zip(xs, ys):
            x, y = transform * (col, row)
            world_xs.append(x)
            world_ys.append(y)

        geom = box(
            min(world_xs),
            min(world_ys),
            max(world_xs),
            max(world_ys),
        )

        records.append(
            {
                "label": det.get("label", "object"),
                "score": float(det.get("score", 0.0)),
                "geometry": geom,
            }
        )

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )

    return gdf


def detection_summary(
    gdf: gpd.GeoDataFrame,
) -> dict:
    """
    Lightweight summary of a detections GeoDataFrame.
    """

    if gdf is None or gdf.empty:
        return {
            "count": 0,
            "labels": {},
            "mean_score": None,
            "n_classes": 0,
        }

    return {
        "count": int(len(gdf)),
        "labels": {
            str(k): int(v)
            for k, v in gdf["label"].value_counts().items()
        },
        "mean_score": float(gdf["score"].mean())
        if "score" in gdf.columns
        else None,
        "n_classes": int(
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