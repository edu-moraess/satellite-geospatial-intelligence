from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EOModelDefinition:
    """Contract for a geospatial foundation/fine-tuned model."""

    model_id: str
    name: str
    task: str
    description: str
    checkpoint: Path | None
    required_bands: tuple[str, ...]
    native_resolution_m: int
    source_domain: str
    classes: tuple[str, ...]
    checkpoint_repo: str | None = None
    input_size: int = 512


MODELS = {
    "prithvi_eo_v2_300m_burn_scars": EOModelDefinition(
        model_id="prithvi_eo_v2_300m_burn_scars",
        name="Prithvi-EO-2.0 300M · Burn Scars",
        task="semantic_segmentation",
        description=(
            "Earth-observation foundation model fine-tuned for pixel-level "
            "burn-scar segmentation. It requires a six-band multispectral "
            "input compatible with the HLS training distribution."
        ),
        checkpoint=None,
        required_bands=("B02", "B03", "B04", "B8A", "B11", "B12"),
        native_resolution_m=30,
        source_domain="HLS",
        classes=("Not burned", "Burn scar"),
        checkpoint_repo="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        input_size=512,
    ),
}


def list_models() -> list[str]:
    return list(MODELS)


def get_model(model_id: str) -> EOModelDefinition:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown deep-learning model: {model_id}") from exc


def checkpoint_available(model_id: str) -> bool:
    checkpoint = get_model(model_id).checkpoint
    return checkpoint is not None and checkpoint.exists()
