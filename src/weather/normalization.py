"""
Pure normalization of Open-Meteo JSON payloads into WeatherObservation
/ WeatherSeries objects.

Deliberately separated from open_meteo.py so these functions can be
unit-tested with fixture dictionaries and no network access
(master prompt item 33).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.weather.models import WeatherObservation, WeatherSeries


def _parse_timestamp(value: str) -> datetime:
    """Parse an Open-Meteo ISO8601 timestamp (hourly or date-only) as UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_current(payload: dict, source: str) -> WeatherObservation:
    """Normalize a /v1/forecast response's `current` block."""
    current = payload.get("current") or {}
    timestamp = (
        _parse_timestamp(current["time"])
        if current.get("time")
        else datetime.now(timezone.utc)
    )

    return WeatherObservation(
        timestamp=timestamp,
        latitude=payload.get("latitude"),
        longitude=payload.get("longitude"),
        temperature=current.get("temperature_2m"),
        apparent_temperature=current.get("apparent_temperature"),
        relative_humidity=current.get("relative_humidity_2m"),
        precipitation=current.get("precipitation"),
        rain=current.get("rain"),
        snowfall=current.get("snowfall"),
        cloud_cover=current.get("cloud_cover"),
        wind_speed=current.get("wind_speed_10m"),
        wind_direction=current.get("wind_direction_10m"),
        wind_gusts=current.get("wind_gusts_10m"),
        surface_pressure=current.get("surface_pressure"),
        source=source,
    )


def _series_from_block(
    payload: dict,
    block_key: str,
    variable_map: dict,
    source: str,
) -> WeatherSeries:
    block = payload.get(block_key) or {}
    times = block.get("time") or []
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")

    observations = []
    for i, time_str in enumerate(times):

        def _value(variable_name):
            values = block.get(variable_name)
            if values is None or i >= len(values):
                return None
            return values[i]

        fields = {
            field_name: _value(variable_name)
            for field_name, variable_name in variable_map.items()
        }

        observations.append(
            WeatherObservation(
                timestamp=_parse_timestamp(time_str),
                latitude=latitude,
                longitude=longitude,
                source=source,
                **fields,
            )
        )

    return WeatherSeries(observations=observations, provider=source)


# Maps WeatherObservation field name -> Open-Meteo hourly variable name.
HOURLY_VARIABLE_MAP = {
    "temperature": "temperature_2m",
    "apparent_temperature": "apparent_temperature",
    "relative_humidity": "relative_humidity_2m",
    "precipitation": "precipitation",
    "rain": "rain",
    "snowfall": "snowfall",
    "cloud_cover": "cloud_cover",
    "wind_speed": "wind_speed_10m",
    "wind_direction": "wind_direction_10m",
    "wind_gusts": "wind_gusts_10m",
    "surface_pressure": "surface_pressure",
    "shortwave_radiation": "shortwave_radiation",
    "soil_temperature": "soil_temperature_0cm",
    "soil_moisture": "soil_moisture_0_to_1cm",
}

# Maps WeatherObservation field name -> Open-Meteo daily variable name.
DAILY_VARIABLE_MAP = {
    "temperature": "temperature_2m_mean",
    "apparent_temperature": "apparent_temperature_mean",
    "precipitation": "precipitation_sum",
    "rain": "rain_sum",
    "snowfall": "snowfall_sum",
    "wind_speed": "wind_speed_10m_max",
    "wind_gusts": "wind_gusts_10m_max",
    "wind_direction": "wind_direction_10m_dominant",
    "shortwave_radiation": "shortwave_radiation_sum",
}


def parse_hourly(payload: dict, source: str) -> WeatherSeries:
    """Normalize a /v1/forecast or /v1/archive response's `hourly` block."""
    return _series_from_block(payload, "hourly", HOURLY_VARIABLE_MAP, source)


def parse_daily(payload: dict, source: str) -> WeatherSeries:
    """Normalize a /v1/forecast or /v1/archive response's `daily` block."""
    return _series_from_block(payload, "daily", DAILY_VARIABLE_MAP, source)
