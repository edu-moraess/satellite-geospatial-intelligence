"""
Weather Intelligence package (master prompt items 6-16).

Importing this package (or any submodule inside it) registers every
built-in provider into src.weather.client.WEATHER_PROVIDERS.
"""

from __future__ import annotations

from src.weather.client import (
    WEATHER_PROVIDERS,
    WeatherProvider,
    WeatherProviderError,
    WeatherProviderUnavailable,
    get_weather_provider,
    list_weather_providers,
)
from src.weather.models import DataQuality, WeatherObservation, WeatherSeries

# Importing these triggers @register_provider on their classes.
from src.weather import open_meteo as _open_meteo  # noqa: F401
from src.weather import openweather as _openweather  # noqa: F401

__all__ = [
    "WEATHER_PROVIDERS",
    "WeatherProvider",
    "WeatherProviderError",
    "WeatherProviderUnavailable",
    "get_weather_provider",
    "list_weather_providers",
    "DataQuality",
    "WeatherObservation",
    "WeatherSeries",
]
