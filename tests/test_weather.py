"""
Tests for the Weather Intelligence layer (master prompt item 33).

None of these hit the network: normalization is tested against
fixture payloads shaped like real Open-Meteo responses, and provider
registry / error-handling behavior is tested through the public
interface only.
"""

from datetime import date, datetime, timezone

import pytest

import src.weather  # noqa: F401  (imports register the built-in providers)
from src.weather.cache import TTLCache, cached_call
from src.weather.client import (
    WEATHER_PROVIDERS,
    WeatherProviderError,
    WeatherProviderUnavailable,
    get_weather_provider,
)
from src.weather.normalization import parse_current, parse_daily, parse_hourly
from src.weather_spatial import get_weather_for_scene


CURRENT_FIXTURE = {
    "latitude": -23.55,
    "longitude": -46.63,
    "current": {
        "time": "2026-08-20T15:00",
        "temperature_2m": 22.4,
        "relative_humidity_2m": 58,
        "apparent_temperature": 21.9,
        "precipitation": 0.0,
        "rain": 0.0,
        "snowfall": 0.0,
        "cloud_cover": 40,
        "surface_pressure": 1015.2,
        "wind_speed_10m": 12.3,
        "wind_direction_10m": 135,
        "wind_gusts_10m": 20.1,
    },
}

HOURLY_FIXTURE = {
    "latitude": -23.55,
    "longitude": -46.63,
    "hourly": {
        "time": ["2026-08-20T00:00", "2026-08-20T01:00", "2026-08-20T02:00"],
        "temperature_2m": [18.1, 17.8, 17.5],
        "relative_humidity_2m": [70, 72, 74],
        "precipitation": [0.0, 0.0, 0.1],
    },
}

DAILY_FIXTURE = {
    "latitude": -23.55,
    "longitude": -46.63,
    "daily": {
        "time": ["2026-08-20", "2026-08-21"],
        "temperature_2m_mean": [19.5, 20.1],
        "precipitation_sum": [0.0, 3.2],
    },
}


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------


def test_parse_current():
    obs = parse_current(CURRENT_FIXTURE, source="open_meteo")
    assert obs.temperature == 22.4
    assert obs.relative_humidity == 58
    assert obs.source == "open_meteo"
    assert obs.latitude == -23.55


def test_parse_hourly_series_length_and_values():
    series = parse_hourly(HOURLY_FIXTURE, source="open_meteo")
    assert len(series) == 3
    assert series.observations[0].temperature == 18.1
    assert series.observations[2].precipitation == 0.1


def test_parse_hourly_missing_variable_is_none():
    series = parse_hourly(HOURLY_FIXTURE, source="open_meteo")
    # wind_speed_10m was never present in this fixture at all.
    assert series.observations[0].wind_speed is None


def test_nearest_observation_picks_closest_timestamp():
    series = parse_hourly(HOURLY_FIXTURE, source="open_meteo")
    target = datetime(2026, 8, 20, 1, 40, tzinfo=timezone.utc)
    nearest = series.nearest(target)
    assert nearest.timestamp.hour == 2  # 20 min from 02:00 vs 40 min from 01:00


def test_nearest_observation_empty_series_returns_none():
    from src.weather.models import WeatherSeries

    assert WeatherSeries(observations=[]).nearest(datetime.now(timezone.utc)) is None


def test_parse_daily():
    series = parse_daily(DAILY_FIXTURE, source="open_meteo")
    assert len(series) == 2
    assert series.observations[1].precipitation == 3.2


# ------------------------------------------------------------
# Provider registry
# ------------------------------------------------------------


def test_provider_registry_contains_built_ins():
    assert "open_meteo" in WEATHER_PROVIDERS
    assert "openweather" in WEATHER_PROVIDERS


def test_unknown_provider_raises():
    with pytest.raises(WeatherProviderError):
        get_weather_provider("not_a_real_provider")


def test_open_meteo_is_always_configured():
    provider = get_weather_provider("open_meteo")
    assert provider.is_configured() is True


def test_openweather_unconfigured_by_default():
    provider = get_weather_provider("openweather")
    assert provider.is_configured() is False
    with pytest.raises(WeatherProviderUnavailable):
        provider.get_current(-23.55, -46.63)


# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------


def test_ttl_cache_hit_avoids_second_call():
    cache = TTLCache(ttl_seconds=60)
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        return "value"

    assert cached_call(cache, ("k",), _fetch) == "value"
    assert cached_call(cache, ("k",), _fetch) == "value"
    assert calls["n"] == 1


def test_ttl_cache_expiry():
    cache = TTLCache(ttl_seconds=-1)  # already expired on arrival
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        return calls["n"]

    assert cached_call(cache, ("k",), _fetch) == 1
    assert cached_call(cache, ("k",), _fetch) == 2  # expired -> fetched again


# ------------------------------------------------------------
# Weather <-> satellite temporal alignment
# ------------------------------------------------------------


def test_get_weather_for_scene_missing_timestamp_is_safe():
    result = get_weather_for_scene(
        latitude=-23.55,
        longitude=-46.63,
        acquisition_datetime=None,
        provider_id="open_meteo",
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["observation"] is None


def test_get_weather_for_scene_unconfigured_provider_is_safe():
    result = get_weather_for_scene(
        latitude=-23.55,
        longitude=-46.63,
        acquisition_datetime=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        provider_id="openweather",
    )
    assert result["status"] == "UNAVAILABLE"
    assert "not configured" in result["reason"]
    assert result["observation"] is None


def test_get_weather_for_scene_unknown_provider_is_safe():
    result = get_weather_for_scene(
        latitude=-23.55,
        longitude=-46.63,
        acquisition_datetime=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        provider_id="not_a_real_provider",
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["observation"] is None


def test_get_weather_for_scene_accepts_date_only():
    # Should not raise even though a `date` has no hour/minute.
    result = get_weather_for_scene(
        latitude=-23.55,
        longitude=-46.63,
        acquisition_datetime=date(2026, 8, 20),
        provider_id="openweather",  # unconfigured -> exercises the date-coercion
        # path without needing a real network call.
    )
    assert result["status"] == "UNAVAILABLE"
