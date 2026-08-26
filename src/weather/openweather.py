"""
OpenWeatherMap provider — registry placeholder (master prompt item 15).

Not implemented yet. Registered so the Weather Source selector can
list "OpenWeather" as a known option and show
"Unavailable — API key required" instead of the app crashing or the
option simply not existing. Implement get_current/get_hourly/get_daily
here when OpenWeather integration is actually prioritized.
"""

from __future__ import annotations

import os
from datetime import date

from src.weather.client import (
    WeatherProvider,
    WeatherProviderUnavailable,
    register_provider,
)
from src.weather.models import WeatherObservation, WeatherSeries


@register_provider
class OpenWeatherProvider(WeatherProvider):
    id = "openweather"
    name = "OpenWeather"
    requires_api_key = True

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _require_configured(self) -> None:
        if not self.is_configured():
            raise WeatherProviderUnavailable(
                "OpenWeather is not configured. Set the OPENWEATHER_API_KEY "
                "environment variable (or st.secrets['OPENWEATHER_API_KEY']) "
                "to enable it."
            )

    def get_current(self, latitude: float, longitude: float) -> WeatherObservation:
        self._require_configured()
        raise NotImplementedError(
            "OpenWeather integration is planned for a later phase "
            "(master prompt Phase 5)."
        )

    def get_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        self._require_configured()
        raise NotImplementedError(
            "OpenWeather integration is planned for a later phase "
            "(master prompt Phase 5)."
        )

    def get_daily(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        self._require_configured()
        raise NotImplementedError(
            "OpenWeather integration is planned for a later phase "
            "(master prompt Phase 5)."
        )
