from __future__ import annotations

from typing import Any

import math


# ============================================================
# VALIDATION
# ============================================================

def _safe_float(
    value: Any,
) -> float | None:

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


# ============================================================
# EXTRACT DRAWINGS
# ============================================================

def extract_drawings(
    map_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Extract geometries created with Folium Draw.

    streamlit-folium returns the drawn objects
    through the 'all_drawings' map state.
    """

    if not map_state:
        return []

    drawings = map_state.get(
        "all_drawings"
    )

    if not drawings:
        return []

    if isinstance(drawings, dict):

        features = drawings.get(
            "features",
            [],
        )

        if isinstance(features, list):
            return features

    if isinstance(drawings, list):
        return drawings

    return []


# ============================================================
# GEOMETRY TYPE
# ============================================================

def geometry_type(
    feature: dict[str, Any],
) -> str:

    geometry = feature.get(
        "geometry",
        {},
    )

    return str(
        geometry.get(
            "type",
            "Unknown",
        )
    )


# ============================================================
# COORDINATES
# ============================================================

def geometry_coordinates(
    feature: dict[str, Any],
) -> list:

    geometry = feature.get(
        "geometry",
        {},
    )

    coordinates = geometry.get(
        "coordinates",
        [],
    )

    return coordinates


# ============================================================
# FLATTEN COORDINATES
# ============================================================

def _flatten_coordinates(
    coordinates: Any,
) -> list[tuple[float, float]]:

    result = []

    if not isinstance(
        coordinates,
        list,
    ):
        return result

    # [lon, lat]
    if (
        len(coordinates) >= 2
        and isinstance(
            coordinates[0],
            (int, float),
        )
        and isinstance(
            coordinates[1],
            (int, float),
        )
    ):

        lon = _safe_float(
            coordinates[0]
        )

        lat = _safe_float(
            coordinates[1]
        )

        if (
            lon is not None
            and lat is not None
        ):

            result.append(
                (
                    lon,
                    lat,
                )
            )

        return result

    for item in coordinates:

        result.extend(
            _flatten_coordinates(
                item
            )
        )

    return result


# ============================================================
# BBOX
# ============================================================

def geometry_bbox(
    feature: dict[str, Any],
) -> tuple[
    float,
    float,
    float,
    float,
] | None:

    coordinates = (
        geometry_coordinates(
            feature
        )
    )

    points = _flatten_coordinates(
        coordinates
    )

    if not points:
        return None

    longitudes = [
        point[0]
        for point in points
    ]

    latitudes = [
        point[1]
        for point in points
    ]

    return (
        min(longitudes),
        min(latitudes),
        max(longitudes),
        max(latitudes),
    )


# ============================================================
# CENTER
# ============================================================

def bbox_center(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[float, float]:

    min_lon, min_lat, max_lon, max_lat = (
        bbox
    )

    return (
        (min_lat + max_lat) / 2,
        (min_lon + max_lon) / 2,
    )


# ============================================================
# APPROXIMATE AREA
# ============================================================

def bbox_area_km2(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    min_lon, min_lat, max_lon, max_lat = (
        bbox
    )

    latitude = (
        min_lat + max_lat
    ) / 2

    latitude_km = (
        111.32
    )

    longitude_km = (
        111.32
        * math.cos(
            math.radians(
                latitude
            )
        )
    )

    width = (
        max_lon - min_lon
    ) * longitude_km

    height = (
        max_lat - min_lat
    ) * latitude_km

    return max(
        width * height,
        0.0,
    )


# ============================================================
# BUILD AOI
# ============================================================

def build_aoi(
    feature: dict[str, Any],
) -> dict[str, Any] | None:

    bbox = geometry_bbox(
        feature
    )

    if bbox is None:
        return None

    center_lat, center_lon = (
        bbox_center(
            bbox
        )
    )

    return {
        "geometry": feature,
        "geometry_type": geometry_type(
            feature
        ),
        "bbox": bbox,
        "center": {
            "latitude": center_lat,
            "longitude": center_lon,
        },
        "area_km2": bbox_area_km2(
            bbox
        ),
    }


# ============================================================
# FIRST VALID AOI
# ============================================================

def get_selected_aoi(
    map_state: dict[str, Any] | None,
) -> dict[str, Any] | None:

    drawings = extract_drawings(
        map_state
    )

    if not drawings:
        return None

    # Prefer polygon/rectangle
    for feature in drawings:

        aoi = build_aoi(
            feature
        )

        if aoi is None:
            continue

        if aoi["geometry_type"] in (
            "Polygon",
            "MultiPolygon",
        ):

            return aoi

    return None


# ============================================================
# FORMAT BBOX
# ============================================================

def format_bbox(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> str:

    min_lon, min_lat, max_lon, max_lat = (
        bbox
    )

    return (
        f"["
        f"{min_lon:.6f}, "
        f"{min_lat:.6f}, "
        f"{max_lon:.6f}, "
        f"{max_lat:.6f}"
        f"]"
    )