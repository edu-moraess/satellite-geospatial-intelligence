"""
Weather <-> Satellite temporal alignment (master prompt item 9).

Answers: "What was the weather condition when this satellite scene
was acquired?"

Rule (master prompt item 34): a weather failure must never break
satellite analysis. get_weather_for_scene() therefore never raises —
any problem (unconfigured provider, network error, no data for the
date range) comes back as a normal dict with status="UNAVAILABLE"
and a human-readable reason, instead of an exception.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.weather.client import (
    WeatherProviderError,
    WeatherProviderUnavailable,
    get_weather_provider,
)


def get_weather_for_scene(
    latitude: float,
    longitude: float,
    acquisition_datetime,
    provider_id: str = "open_meteo",
    window_hours: int = 6,
) -> dict:
    """
    Return weather context for a satellite scene acquisition.

    Returns a dict:
        {
            "status": "AVAILABLE" | "UNAVAILABLE",
            "provider": provider_id,
            "reason": str | None,
            "observation": WeatherObservation | None,
            "hours_from_acquisition": float | None,
        }
    """

    result = {
        "status": "UNAVAILABLE",
        "provider": provider_id,
        "reason": None,
        "observation": None,
        "hours_from_acquisition": None,
        "series": None,
    }

    if acquisition_datetime is None:
        result["reason"] = "Scene has no acquisition timestamp."
        return result

    try:
        provider = get_weather_provider(provider_id)
    except WeatherProviderError as error:
        result["reason"] = str(error)
        return result

    if not provider.is_configured():
        result["reason"] = f"{provider.name} is not configured."
        return result

    if isinstance(acquisition_datetime, datetime):
        acquisition = acquisition_datetime
    else:
        # Accept a date-only value (e.g. item.datetime.date()) too.
        acquisition = datetime(
            acquisition_datetime.year,
            acquisition_datetime.month,
            acquisition_datetime.day,
            12,
            0,
        )

    if acquisition.tzinfo is None:
        acquisition = acquisition.replace(tzinfo=timezone.utc)

    start_date = (acquisition - timedelta(hours=window_hours)).date()
    end_date = (acquisition + timedelta(hours=window_hours)).date()

    try:
        series = provider.get_hourly(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )
    except WeatherProviderUnavailable as error:
        result["reason"] = str(error)
        return result
    except WeatherProviderError as error:
        result["reason"] = f"{provider.name} request failed: {error}"
        return result
    except Exception as error:  # noqa: BLE001 - never let weather break satellite
        result["reason"] = f"{provider.name} request failed: {error}"
        return result

    observation = series.nearest(acquisition)

    if observation is None:
        result["reason"] = "No weather observation found for this time window."
        return result

    gap_hours = abs((observation.timestamp - acquisition).total_seconds()) / 3600.0

    result.update(
        {
            "status": "AVAILABLE",
            "reason": None,
            "observation": observation,
            "hours_from_acquisition": round(gap_hours, 2),
            "series": series,
        }
    )
    return result
