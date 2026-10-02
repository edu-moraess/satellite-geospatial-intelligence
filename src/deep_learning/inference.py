from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.deep_learning.input_adapter import (
    InputCompatibility,
    check_multispectral_input,
)
from src.deep_learning.registry import (
    checkpoint_available,
    get_model,
)


@dataclass(frozen=True)
class InferenceGate:
    ready: bool
    compatibility: InputCompatibility
    reason: str


def build_inference_gate(
    model_id: str,
    available_bands: set[str] | list[str] | tuple[str, ...],
    source_domain: str = "sentinel2-l2a",
) -> InferenceGate:
    """Return an explicit readiness gate; never fall back to synthetic output."""
    model = get_model(model_id)
    compatibility = check_multispectral_input(available_bands, model.required_bands)

    if not compatibility.ready:
        return InferenceGate(False, compatibility, compatibility.message)

    if model.source_domain.lower() != source_domain.lower():
        return InferenceGate(
            False,
            compatibility,
            (
                "Deep-learning inference blocked: the registered checkpoint "
                f"expects {model.source_domain} inputs, but the current scene "
                f"is {source_domain}. A harmonization adapter is required; "
                "simple band renaming/resampling is not sufficient."
            ),
        )

    if not checkpoint_available(model_id):
        return InferenceGate(
            False,
            compatibility,
            "Deep-learning inference blocked: no real checkpoint is registered.",
        )

    return InferenceGate(
        True,
        compatibility,
        "Model checkpoint and input contract are ready.",
    )


def validate_cube(cube: Any, expected_bands: int) -> np.ndarray:
    """Validate a CxHxW multispectral tensor before handing it to a model."""
    array = np.asarray(cube, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Expected multispectral cube with shape C x H x W.")
    if array.shape[0] != expected_bands:
        raise ValueError(
            f"Expected {expected_bands} bands, received {array.shape[0]}."
        )
    if not np.isfinite(array).any():
        raise ValueError("Multispectral cube contains no valid pixels.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
