"""VYRA SmallCNN B07 LULC field-test contract.

This module deliberately separates the frozen training identity from the
satellite application's runtime. The network architecture is not recreated
here: the GATE 8 repository does not publish the training implementation or
checkpoint bytes. Inference remains blocked until those exact artifacts are
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


EXPECTED_CHECKPOINT_SHA256 = (
    "ef56a68ad7362d4bb1604b29c5669ac27851284bd15cb56468a036b1c4d90c2a"
)
PATCH_FINGERPRINT = (
    "e5ad2d547726a22a8a069247946f099be293c7d7f6488924551c927b240b2d72"
)
EXPERIMENT_FINGERPRINT = (
    "beaecb493ad09e6cb60740447053dffa20c4019e904cb5906c335252b7b1332f"
)

PATCH_SIZE = 64
INPUT_CHANNELS = 1
BAND = "B07"
STAC_ASSET_KEY = "rededge3"
GSD_M = 20
PATCH_SIZE_M = 1280


@dataclass(frozen=True)
class VyraInferenceContract:
    model_id: str = "vyra_smallcnn_b07_lulc_v1"
    architecture: str = "SmallCNN"
    input_shape: tuple[int, int, int] = (1, 64, 64)
    classes: int = 29
    band: str = BAND
    stac_asset_key: str = STAC_ASSET_KEY
    patch_size: int = PATCH_SIZE
    stride: int = PATCH_SIZE
    gsd_m: int = GSD_M
    patch_size_m: int = PATCH_SIZE_M
    augmentation: str = "none"
    checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256
    patch_fingerprint: str = PATCH_FINGERPRINT
    experiment_fingerprint: str = EXPERIMENT_FINGERPRINT


CONTRACT = VyraInferenceContract()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VYRA checkpoint not found: {path}")
    actual = sha256_file(path)
    if actual != CONTRACT.checkpoint_sha256:
        raise ValueError(
            "VYRA checkpoint identity mismatch: "
            f"expected {CONTRACT.checkpoint_sha256}, got {actual}"
        )


def prepare_b07_patch(array: np.ndarray) -> np.ndarray:
    """Prepare one already-windowed B07 patch without inventing normalization."""
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D B07 patch, got shape={arr.shape}")
    if arr.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"Expected {(PATCH_SIZE, PATCH_SIZE)}, got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("B07 patch contains non-finite values")

    # GATE 8 records no augmentation. It does not publish a normalization
    # contract, so this function intentionally performs no scaling/clipping.
    return arr.astype(np.float32, copy=False)[None, :, :]


def load_vyra_model(*_args, **_kwargs):
    """Refuse unsafe reconstruction until the exact frozen model is available."""
    raise RuntimeError(
        "VYRA inference is BLOCKED: GATE 8 identifies the frozen SmallCNN "
        "checkpoint, but the exact training implementation/checkpoint bytes "
        "are not available in this repository. Do not reconstruct the network "
        "from parameter count. Supply the exact checkpoint artifact or the "
        "original model definition before enabling predictions."
    )
