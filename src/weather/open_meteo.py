"""
Open-Meteo weather provider (master prompt item 7).

Free, global coverage, no API key required for non-commercial use.

Variable names below were checked against https://open-meteo.com/en/docs
on 2026-08-26. Open-Meteo adds/renames variables over time; if a
request starts failing with "Cannot initialize WeatherVariable from
invalid String value ...", re-check that page before assuming
something else is wrong.

Two endpoints are used:
  - FORECAST_URL (/v1/forecast): current conditions, forecast, and
    recent history via `start_date`/`end_date` — but only reaches
    back `past_days` (max 92) before the request date.
  - ARCHIVE_URL (/v1/archive): the ERA5-backed Historical Weather API,
    used automatically for dates older than ~80 days so satellite
    scenes from earlier in the year (or in past years) still resolve
    to a weather observation instead of failing.
"""

from __future__ import annotations

from datetime import date

import requests

from src.weather.cache import TTLCache, cached_call
from src.weather.client import (
    WeatherProvider,
    WeatherProviderError,
    register_provider,
)
from src.weather.models import WeatherObservation, WeatherSeries
from src.weather.normalization import parse_current, parse_daily, parse_hourly


@register_provider
class OpenMeteoProvider(WeatherProvider):
    id = "open_meteo"
    name = "Open-Meteo"
    requires_api_key = False

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Kept out of `current` where Open-Meteo doesn't expose them (soil /
    # radiation variables are hourly-only, per the docs page above).
    CURRENT_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "showers",
        "snowfall",
        "cloud_cover",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]

    HOURLY_VARIABLES = CURRENT_VARIABLES + [
        "shortwave_radiation",
        "soil_temperature_0cm",
        "soil_moisture_0_to_1cm",
    ]

    DAILY_VARIABLES = [
        "temperature_2m_mean",
        "apparent_temperature_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
    ]

    # Forecast API's `past_days` caps at 92; stay comfortably under that
    # before switching to the historical/archive endpoint.
    _ARCHIVE_THRESHOLD_DAYS = 80

    def __init__(self, timeout_seconds: float = 10.0, cache: TTLCache | None = None):
        self.timeout_seconds = timeout_seconds
        self._cache = cache or TTLCache(ttl_seconds=900.0)

    def is_configured(self) -> bool:
        return True

    # ------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------

    def _common_params(self, latitude: float, longitude: float) -> dict:
        return {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timeformat": "iso8601",
        }

    def _request(self, url: str, params: dict) -> dict:
        try:
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
        except requests.RequestException as error:
            raise WeatherProviderError(
                f"Network error contacting Open-Meteo: {error}"
            ) from error

        if response.status_code != 200:
            reason = None
            try:
                reason = response.json().get("reason")
            except ValueError:
                pass
            raise WeatherProviderError(
                f"Open-Meteo returned HTTP {response.status_code}"
                + (f": {reason}" if reason else ".")
            )

        try:
            return response.json()
        except ValueError as error:
            raise WeatherProviderError(
                f"Open-Meteo returned invalid JSON: {error}"
            ) from error

    def _hourly_url_for(self, start_date: date) -> str:
        if (date.today() - start_date).days > self._ARCHIVE_THRESHOLD_DAYS:
            return self.ARCHIVE_URL
        return self.FORECAST_URL

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def get_current(self, latitude: float, longitude: float) -> WeatherObservation:
        params = self._common_params(latitude, longitude)
        params["current"] = ",".join(self.CURRENT_VARIABLES)

        cache_key = ("current", round(latitude, 4), round(longitude, 4))

        def _fetch():
            payload = self._request(self.FORECAST_URL, params)
            return parse_current(payload, source=self.id)

        return cached_call(self._cache, cache_key, _fetch)

    def get_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        url = self._hourly_url_for(start_date)

        params = self._common_params(latitude, longitude)
        params["hourly"] = ",".join(self.HOURLY_VARIABLES)
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()

        cache_key = (
            "hourly",
            url,
            round(latitude, 4),
            round(longitude, 4),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        def _fetch():
            payload = self._request(url, params)
            return parse_hourly(payload, source=self.id)

        return cached_call(self._cache, cache_key, _fetch)

    def get_daily(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        url = self._hourly_url_for(start_date)

        params = self._common_params(latitude, longitude)
        params["daily"] = ",".join(self.DAILY_VARIABLES)
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()

        cache_key = (
            "daily",
            url,
            round(latitude, 4),
            round(longitude, 4),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        def _fetch():
            payload = self._request(url, params)
            return parse_daily(payload, source=self.id)

        return cached_call(self._cache, cache_key, _fetch)

    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        params = self._common_params(latitude, longitude)
        params["hourly"] = ",".join(self.HOURLY_VARIABLES)
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()

        cache_key = (
            "historical",
            round(latitude, 4),
            round(longitude, 4),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        def _fetch():
            payload = self._request(self.ARCHIVE_URL, params)
            return parse_hourly(payload, source=self.id)

        return cached_call(self._cache, cache_key, _fetch)
