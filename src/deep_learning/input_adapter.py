from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputCompatibility:
    ready: bool
    missing_bands: tuple[str, ...]
    extra_bands: tuple[str, ...]
    message: str


def check_multispectral_input(
    available_bands: set[str] | list[str] | tuple[str, ...],
    required_bands: tuple[str, ...],
) -> InputCompatibility:
    """Validate the band contract before model inference.

    This deliberately does not silently substitute B08 for B8A or fabricate
    a missing SWIR2 channel. A foundation model must receive the distribution
    it was designed/fine-tuned for.
    """
    available = set(available_bands)
    required = set(required_bands)
    missing = tuple(sorted(required - available))
    extra = tuple(sorted(available - required))

    if missing:
        return InputCompatibility(
            ready=False,
            missing_bands=missing,
            extra_bands=extra,
            message=(
                "Deep-learning inference blocked: required multispectral "
                f"bands are missing ({', '.join(missing)})."
            ),
        )

    return InputCompatibility(
        ready=True,
        missing_bands=(),
        extra_bands=extra,
        message="Multispectral input satisfies the model band contract.",
    )
