"""
Weather Provider abstraction layer (master prompt items 6/15).

Nothing outside this package should ever import OpenMeteoProvider or
OpenWeatherProvider directly. All callers go through
get_weather_provider(provider_id) so the concrete data source can be
swapped or extended (Meteomatics, Google Earth Engine, ...) without
touching weather_spatial.py, ui/, or app.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from src.weather.models import WeatherObservation, WeatherSeries


class WeatherProviderError(Exception):
    """A provider could not fulfil a request (network, bad response, ...)."""


class WeatherProviderUnavailable(WeatherProviderError):
    """A provider exists but is not configured (e.g. missing API key)."""


class WeatherProvider(ABC):
    """Common interface every weather data source must implement."""

    id: str = "base"
    name: str = "Base Provider"
    requires_api_key: bool = False

    @abstractmethod
    def is_configured(self) -> bool:
        """True when this provider is ready to be called right now."""

    @abstractmethod
    def get_current(self, latitude: float, longitude: float) -> WeatherObservation:
        """Current weather conditions for a coordinate."""

    @abstractmethod
    def get_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        """Hourly weather series covering [start_date, end_date] (inclusive)."""

    @abstractmethod
    def get_daily(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        """Daily-aggregated weather series covering [start_date, end_date]."""

    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WeatherSeries:
        """
        Historical weather for a past date range.

        Default implementation delegates to get_hourly(). Providers
        with a dedicated archive/historical endpoint (e.g. Open-Meteo's
        ERA5-backed Historical Weather API) should override this.
        """
        return self.get_hourly(latitude, longitude, start_date, end_date)


# Populated by @register_provider as each provider module is imported.
# Importing `src.weather` (the package __init__) guarantees every
# built-in provider has registered itself.
WEATHER_PROVIDERS: dict = {}


def register_provider(provider_cls):
    """Class decorator: adds a provider class to WEATHER_PROVIDERS."""
    WEATHER_PROVIDERS[provider_cls.id] = provider_cls
    return provider_cls


def list_weather_providers() -> list:
    """Ids of every registered provider (configured or not)."""
    return list(WEATHER_PROVIDERS.keys())


def get_weather_provider(provider_id: str = "open_meteo") -> WeatherProvider:
    """
    Instantiate a weather provider by id.

    Raises WeatherProviderError if the id is unknown to the registry.
    Does NOT raise if the provider exists but lacks an API key —
    check provider.is_configured() for that, so the UI can show
    "Unavailable — API key required" instead of crashing
    (master prompt item 15).
    """
    provider_cls = WEATHER_PROVIDERS.get(provider_id)
    if provider_cls is None:
        raise WeatherProviderError(
            f"Unknown weather provider: '{provider_id}'. "
            f"Available: {list_weather_providers()}"
        )
    return provider_cls()
