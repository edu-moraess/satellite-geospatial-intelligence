from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelDefinition:

    name: str
    description: str
    checkpoint: Path | None
    classes: tuple[str, ...]
    input_size: int


AVAILABLE_MODELS = {

    "remote_sensing_baseline": ModelDefinition(
        name="Remote Sensing Baseline",
        description=(
            "Interface preparada para um "
            "checkpoint específico de sensoriamento remoto."
        ),
        checkpoint=None,
        classes=(
            "Buildings",
            "Roads",
            "Vehicles",
            "Aircraft",
            "Ships",
            "Storage Tanks",
        ),
        input_size=512,
    ),

}


def get_model(
    model_id: str,
) -> ModelDefinition:

    if model_id not in AVAILABLE_MODELS:

        raise KeyError(
            f"Unknown model: {model_id}"
        )

    return AVAILABLE_MODELS[
        model_id
    ]


def list_models() -> list[str]:

    return list(
        AVAILABLE_MODELS.keys()
    )


def model_available(
    model_id: str,
) -> bool:

    model = get_model(
        model_id
    )

    if model.checkpoint is None:

        return False

    return model.checkpoint.exists()