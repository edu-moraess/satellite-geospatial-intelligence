"""Runtime inference for the official Prithvi-EO-2.0 Burn Scars checkpoint.

Dependencies are imported lazily because the 1.3 GB checkpoint and
TerraTorch/PyTorch stack are intentionally not part of the base GEOCORE
Streamlit runtime yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
CONFIG_FILE = "burn_scars_config.yaml"
CHECKPOINT_FILE = "Prithvi_EO_V2_300M_BurnScars.pt"


@dataclass(frozen=True)
class BurnScarPrediction:
    mask: np.ndarray
    burned_fraction: float
    valid_fraction: float
    model_id: str
    checkpoint_repo: str


def runtime_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import terratorch  # noqa: F401
        import huggingface_hub  # noqa: F401
        import einops  # noqa: F401
    except ImportError as exc:
        return False, (
            "Prithvi runtime dependencies are not installed. "
            f"Missing: {exc.name}. Install requirements-deep-learning.txt."
        )
    return True, "Prithvi runtime dependencies available."


@lru_cache(maxsize=1)
def _load_model():
    from huggingface_hub import hf_hub_download
    from terratorch.cli_tools import LightningInferenceModel

    config = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CONFIG_FILE,
    )
    checkpoint = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CHECKPOINT_FILE,
    )
    loaded = LightningInferenceModel.from_config(config, checkpoint)
    loaded.model.eval()
    return loaded


def predict_burn_scars(
    cube: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    input_size: int = 512,
) -> BurnScarPrediction:
    """Run the official Burn Scars checkpoint on a normalized CxHxW cube."""
    available, reason = runtime_available()
    if not available:
        raise RuntimeError(reason)

    from einops import rearrange
    import torch

    array = np.asarray(cube, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 6:
        raise ValueError("Expected a six-band CxHxW Prithvi cube.")

    height, width = array.shape[-2:]
    pad_h = (input_size - height % input_size) % input_size
    pad_w = (input_size - width % input_size) % input_size
    padded = np.pad(
        array,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )

    windows = torch.from_numpy(padded).unsqueeze(0)
    windows = windows.unfold(2, input_size, input_size).unfold(
        3, input_size, input_size
    )
    h_tiles, w_tiles = windows.shape[2:4]
    windows = rearrange(
        windows,
        "b c h1 w1 h w -> (b h1 w1) c h w",
        h=input_size,
        w=input_size,
    )

    model = _load_model()
    predictions = []

    with torch.no_grad():
        for window in windows:
            sample = window.numpy().transpose(1, 2, 0)
            transformed = model.datamodule.test_transform(image=sample)
            image = transformed["image"].unsqueeze(0)
            image = model.datamodule.aug({"image": image})["image"]
            image = image.to(model.device)
            logits = model.model(image).output
            predictions.append(logits.argmax(dim=1).cpu())

    pred = torch.stack(predictions)
    pred = rearrange(
        pred,
        "(b h1 w1) h w -> b (h1 h) (w1 w)",
        b=1,
        h1=h_tiles,
        w1=w_tiles,
        h=input_size,
        w=input_size,
    )[0].numpy()

    pred = pred[:height, :width].astype(np.uint8, copy=False)

    if valid_mask is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError("valid_mask must match the spatial cube shape.")

    burned = (pred == 1) & valid
    valid_pixels = int(valid.sum())
    burned_fraction = float(burned.sum() / valid_pixels) if valid_pixels else 0.0

    return BurnScarPrediction(
        mask=pred,
        burned_fraction=burned_fraction,
        valid_fraction=float(valid_pixels / valid.size) if valid.size else 0.0,
        model_id="prithvi_eo_v2_300m_burn_scars",
        checkpoint_repo=MODEL_REPO,
    )
