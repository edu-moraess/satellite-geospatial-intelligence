from __future__ import annotations

import math
from typing import Any


# ============================================================
# VALIDATION
# ============================================================

def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_aoi_center(
    latitude: Any,
    longitude: Any,
) -> tuple[float, float] | None:
    """
    Validate and normalize an AOI center.

    Returns:
        (latitude, longitude) when valid.
        None when invalid.
    """

    lat = _safe_float(latitude)
    lon = _safe_float(longitude)

    if lat is None or lon is None:
        return None

    if not math.isfinite(lat) or not math.isfinite(lon):
        return None

    if lat < -90.0 or lat > 90.0:
        return None

    if lon < -180.0 or lon > 180.0:
        return None

    return lat, lon


def validate_area_size(
    area_size: Any,
    minimum: float = 0.001,
    maximum: float = 20.0,
) -> float | None:
    """
    Validate AOI area-size parameter expressed in degrees.

    The application UI normally restricts this to a much
    smaller range, but this helper keeps the state safe.
    """

    size = _safe_float(area_size)

    if size is None:
        return None

    if not math.isfinite(size):
        return None

    if size < minimum or size > maximum:
        return None

    return size


# ============================================================
# MAP CLICK → AOI STATE
# ============================================================

def apply_map_click_to_center(
    latitude: Any,
    longitude: Any,
    current_latitude: Any = None,
    current_longitude: Any = None,
    tolerance: float = 1e-7,
) -> tuple[float, float] | None:
    """
    Validate a map click and determine whether it represents
    a new AOI center.

    This function performs no Streamlit I/O and does not modify
    session_state directly.

    Returns:
        (lat, lon) when a valid new center is available.
        None when the click is invalid or effectively unchanged.
    """

    clicked = validate_aoi_center(
        latitude,
        longitude,
    )

    if clicked is None:
        return None

    lat, lon = clicked

    current = validate_aoi_center(
        current_latitude,
        current_longitude,
    )

    if current is not None:
        current_lat, current_lon = current

        if (
            abs(lat - current_lat) <= tolerance
            and abs(lon - current_lon) <= tolerance
        ):
            return None

    return lat, lon


# ============================================================
# EXTRACT DRAWINGS
# ============================================================

def extract_drawings(
    map_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Extract geometries created with Folium Draw.

    streamlit-folium returns drawn objects through
    the 'all_drawings' map-state field.
    """

    if not map_state:
        return []

    drawings = map_state.get("all_drawings")

    if not drawings:
        return []

    if isinstance(drawings, dict):
        features = drawings.get("features", [])

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

    geometry = feature.get("geometry", {})

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

    geometry = feature.get("geometry", {})

    return geometry.get(
        "coordinates",
        [],
    )


# ============================================================
# FLATTEN COORDINATES
# ============================================================

def _flatten_coordinates(
    coordinates: Any,
) -> list[tuple[float, float]]:

    result: list[tuple[float, float]] = []

    if not isinstance(coordinates, list):
        return result

    # GeoJSON coordinate pair:
    # [longitude, latitude]
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
        lon = _safe_float(coordinates[0])
        lat = _safe_float(coordinates[1])

        if lon is not None and lat is not None:
            result.append(
                (
                    lon,
                    lat,
                )
            )

        return result

    for item in coordinates:
        result.extend(
            _flatten_coordinates(item)
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

    coordinates = geometry_coordinates(feature)

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

    min_lon, min_lat, max_lon, max_lat = bbox

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

    min_lon, min_lat, max_lon, max_lat = bbox

    latitude = (
        min_lat + max_lat
    ) / 2

    latitude_km = 111.32

    longitude_km = (
        111.32
        * math.cos(
            math.radians(latitude)
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

    bbox = geometry_bbox(feature)

    if bbox is None:
        return None

    center_lat, center_lon = bbox_center(bbox)

    return {
        "geometry": feature,
        "geometry_type": geometry_type(feature),
        "bbox": bbox,
        "center": {
            "latitude": center_lat,
            "longitude": center_lon,
        },
        "area_km2": bbox_area_km2(bbox),
    }


# ============================================================
# FIRST VALID AOI
# ============================================================

def get_selected_aoi(
    map_state: dict[str, Any] | None,
) -> dict[str, Any] | None:

    drawings = extract_drawings(map_state)

    if not drawings:
        return None

    # Prefer polygon / multipolygon geometries.
    for feature in drawings:
        aoi = build_aoi(feature)

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

    min_lon, min_lat, max_lon, max_lat = bbox

    return (
        f"["
        f"{min_lon:.6f}, "
        f"{min_lat:.6f}, "
        f"{max_lon:.6f}, "
        f"{max_lat:.6f}"
        f"]"
    )