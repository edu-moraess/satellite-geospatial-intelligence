"""Real Prithvi-EO-2.0 Burn Scars inference.

The execution path mirrors the official model repository's inference.py:
LightningInferenceModel.from_config, 512-pixel sliding windows and the
checkpoint-provided test transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
CONFIG_FILE = "burn_scars_config.yaml"
CHECKPOINT_FILE = "Prithvi_EO_V2_300M_BurnScars.pt"
INPUT_SIZE = 512


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
        import yaml  # noqa: F401
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

    config = hf_hub_download(repo_id=MODEL_REPO, filename=CONFIG_FILE)
    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)
    model = LightningInferenceModel.from_config(config, checkpoint)
    model.model.eval()
    return model


def predict_burn_scars(
    cube: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> BurnScarPrediction:
    available, reason = runtime_available()
    if not available:
        raise RuntimeError(reason)

    import torch
    from einops import rearrange

    array = np.asarray(cube, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 6:
        raise ValueError("Expected six-band CxHxW Prithvi input.")

    height, width = array.shape[-2:]
    if float(np.nanmax(array)) > 1.0:
        array = array / 10000.0
    array = np.nan_to_num(array, nan=0.0001, posinf=0.0001, neginf=0.0001)

    # Official inference layout: B x C x T x H x W.
    input_data = array[None, :, None, :, :]
    pad_h = (INPUT_SIZE - height % INPUT_SIZE) % INPUT_SIZE
    pad_w = (INPUT_SIZE - width % INPUT_SIZE) % INPUT_SIZE
    padded = np.pad(
        input_data,
        ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )

    batch = torch.tensor(padded, device="cpu")
    windows = batch.unfold(3, INPUT_SIZE, INPUT_SIZE).unfold(
        4, INPUT_SIZE, INPUT_SIZE
    )
    h_tiles, w_tiles = windows.shape[3:5]
    windows = rearrange(
        windows,
        "b c t h1 w1 h w -> (b h1 w1) c t h w",
        h=INPUT_SIZE,
        w=INPUT_SIZE,
    )

    model = _load_model()
    predictions = []

    for window in torch.tensor_split(windows, max(1, windows.shape[0]), dim=0):
        x = model.datamodule.test_transform(
            image=window.squeeze().numpy().transpose(1, 2, 0)
        )
        x["image"] = x["image"].unsqueeze(0)
        x = model.datamodule.aug(x)["image"].to(model.device)
        with torch.no_grad():
            pred = model.model(x).output.detach().cpu()
        predictions.append(pred.argmax(dim=1))

    pred = torch.cat(predictions, dim=0)
    pred = rearrange(
        pred,
        "(b h1 w1) h w -> b 1 (h1 h) (w1 w)",
        b=1,
        h1=h_tiles,
        w1=w_tiles,
        h=INPUT_SIZE,
        w=INPUT_SIZE,
    )[0, 0].numpy()
    pred = pred[:height, :width].astype(np.uint8, copy=False)

    if valid_mask is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError("valid_mask must match the cube spatial shape.")

    valid_pixels = int(valid.sum())
    burned = (pred == 1) & valid
    fraction = float(burned.sum() / valid_pixels) if valid_pixels else 0.0

    return BurnScarPrediction(
        mask=pred,
        burned_fraction=fraction,
        valid_fraction=float(valid_pixels / valid.size) if valid.size else 0.0,
        model_id="prithvi_eo_v2_300m_burn_scars",
        checkpoint_repo=MODEL_REPO,
    )
