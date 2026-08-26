"""
Normalized weather data models.

Every WeatherProvider (Open-Meteo, OpenWeather, ...) returns data
shaped as WeatherObservation / WeatherSeries, regardless of the
provider's own JSON schema. Nothing downstream (weather_spatial.py,
ui/weather_panel.py, risk.py) should ever touch a provider's raw
response dict directly (master prompt item 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# Units are fixed by the normalization layer (src/weather/open_meteo.py
# always requests Celsius / km/h / mm), so every WeatherObservation is
# guaranteed to be in these units regardless of provider.
UNITS = {
    "temperature": "°C",
    "apparent_temperature": "°C",
    "relative_humidity": "%",
    "precipitation": "mm",
    "rain": "mm",
    "snowfall": "cm",
    "cloud_cover": "%",
    "wind_speed": "km/h",
    "wind_direction": "°",
    "wind_gusts": "km/h",
    "surface_pressure": "hPa",
    "shortwave_radiation": "W/m²",
    "soil_temperature": "°C",
    "soil_moisture": "m³/m³",
}


@dataclass
class WeatherObservation:
    """
    A single normalized weather record (instant or daily-aggregated).

    Any field may be None: not every provider/endpoint/location
    returns every variable. Consumers must handle missing values —
    never assume completeness (master prompt item 12/50).
    """

    timestamp: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    temperature: Optional[float] = None
    apparent_temperature: Optional[float] = None
    relative_humidity: Optional[float] = None
    precipitation: Optional[float] = None
    rain: Optional[float] = None
    snowfall: Optional[float] = None
    cloud_cover: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    wind_gusts: Optional[float] = None
    surface_pressure: Optional[float] = None
    shortwave_radiation: Optional[float] = None
    soil_temperature: Optional[float] = None
    soil_moisture: Optional[float] = None

    source: str = "unknown"


@dataclass
class WeatherSeries:
    """A time-ordered collection of WeatherObservation from one provider."""

    observations: list = field(default_factory=list)
    provider: str = "unknown"

    def __len__(self) -> int:
        return len(self.observations)

    def nearest(self, when: datetime) -> Optional[WeatherObservation]:
        """
        Return the observation closest in time to `when`.

        Returns None when the series is empty. Both `when` and every
        observation timestamp must be timezone-aware for a meaningful
        comparison; naive datetimes are treated as UTC by the callers
        in this package (see weather_spatial.get_weather_for_scene).
        """
        if not self.observations:
            return None
        return min(
            self.observations,
            key=lambda obs: abs((obs.timestamp - when).total_seconds()),
        )


@dataclass
class DataQuality:
    """
    Source transparency metadata (master prompt item 13).

    status is one of: AVAILABLE | PARTIAL | UNAVAILABLE.
    """

    source: str
    spatial_resolution: Optional[str] = None
    temporal_resolution: Optional[str] = None
    coverage: Optional[str] = None
    latency: Optional[str] = None
    status: str = "AVAILABLE"
    reason: Optional[str] = None
