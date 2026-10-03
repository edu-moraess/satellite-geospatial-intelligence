"""Dataset manifest contracts for the VYRA/GEOCORE land-cover dataset.

The manifest keeps land-cover labels separate from geographic domain metadata.
WorldCover source codes remain lossless; the existing five-class GEOCORE mapping
is recorded separately when available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Iterable

WORLD_COVER_CLASSES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow/ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss/lichen",
}

BIOME_DOMAINS = (
    "Amazon",
    "Cerrado",
    "Caatinga",
    "Pantanal",
    "Atlantic Forest",
    "Temperate Forest",
    "Tundra",
    "Arid",
    "Coastal",
    "Wetland",
    "Agricultural",
    "Urban",
)

CLIMATE_DOMAINS = (
    "Tropical",
    "Subtropical",
    "Semi-arid",
    "Arid",
    "Temperate",
    "Mediterranean",
    "Polar",
)

REQUIRED_FIELDS = (
    "sample_id",
    "source",
    "scene_id",
    "latitude",
    "longitude",
    "biome",
    "climate_domain",
    "worldcover_class",
    "acquisition_date",
    "spatial_block",
)


@dataclass(frozen=True)
class DatasetSample:
    """One auditable geospatial sample in the VYRA/GEOCORE manifest."""

    sample_id: str
    source: str
    scene_id: str
    latitude: float
    longitude: float
    biome: str
    climate_domain: str
    worldcover_class: int
    acquisition_date: str
    spatial_block: str
    country: str | None = None
    continent: str | None = None
    geocore_class: int | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")
        if self.biome not in BIOME_DOMAINS:
            raise ValueError(f"Unsupported biome domain: {self.biome!r}")
        if self.climate_domain not in CLIMATE_DOMAINS:
            raise ValueError(f"Unsupported climate domain: {self.climate_domain!r}")
        if self.worldcover_class not in WORLD_COVER_CLASSES:
            raise ValueError(
                f"Unsupported WorldCover class code: {self.worldcover_class}"
            )
        try:
            date.fromisoformat(self.acquisition_date[:10])
        except ValueError as exc:
            raise ValueError(
                "acquisition_date must start with an ISO-8601 date"
            ) from exc
        if not self.spatial_block.strip():
            raise ValueError("spatial_block must be non-empty")
        if self.geocore_class is not None and not 0 <= self.geocore_class <= 4:
            raise ValueError("geocore_class must be between 0 and 4")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest record."""
        return asdict(self)


def validate_manifest(records: Iterable[DatasetSample]) -> None:
    """Validate uniqueness and required invariants across manifest records."""
    seen_ids: set[str] = set()
    seen_blocks: set[str] = set()

    for record in records:
        if record.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        seen_blocks.add(record.spatial_block)

    if not seen_ids:
        raise ValueError("Manifest cannot be empty")


def write_manifest(records: Iterable[DatasetSample], path: str | Path) -> Path:
    """Write deterministic newline-delimited JSON manifest records."""
    materialized = list(records)
    validate_manifest(materialized)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in sorted(materialized, key=lambda item: item.sample_id):
            handle.write(
                json.dumps(record.to_record(), sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    return output


def read_manifest(path: str | Path) -> list[DatasetSample]:
    """Read and validate a newline-delimited JSON manifest."""
    records: list[DatasetSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(DatasetSample(**payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid manifest record at line {line_number}"
                ) from exc
    validate_manifest(records)
    return records
