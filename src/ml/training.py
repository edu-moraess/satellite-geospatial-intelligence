"""Offline LightGBM training and evaluation utilities for experimental GEOCORE ML."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.features import FeatureResult
from src.ml.labels import CLASS_NAMES

RANDOM_SEED = 42
N_CLASSES = len(CLASS_NAMES)


def spatial_block_split(
    shape: tuple[int, int],
    *,
    block_size: int = 32,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Split a raster into non-overlapping spatial blocks.

    Every pixel in a block belongs to exactly one split. Fractions are
    approximate because complete blocks are the unit of assignment.
    """
    if len(shape) != 2 or any(int(v) <= 0 for v in shape):
        raise ValueError(f"shape must contain two positive dimensions, got {shape!r}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    rows, cols = shape
    blocks: list[tuple[int, int]] = [
        (r, c)
        for r in range(0, rows, block_size)
        for c in range(0, cols, block_size)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)

    n = len(blocks)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = int(round(n * validation_fraction))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    split_names = (
        ("train", blocks[:n_train]),
        ("validation", blocks[n_train : n_train + n_val]),
        ("test", blocks[n_train + n_val :]),
    )
    masks = {name: np.zeros(shape, dtype=bool) for name, _ in split_names}

    for name, assigned in split_names:
        for row, col in assigned:
            masks[name][row : row + block_size, col : col + block_size] = True

    if np.any(masks["train"] & masks["validation"]):
        raise AssertionError("Spatial train/validation blocks overlap")
    if np.any(masks["train"] & masks["test"]):
        raise AssertionError("Spatial train/test blocks overlap")
    if np.any(masks["validation"] & masks["test"]):
        raise AssertionError("Spatial validation/test blocks overlap")

    return masks


def prepare_split(
    features: FeatureResult,
    labels: np.ndarray,
    split_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract valid feature rows and labels for one spatial split."""
    labels = np.asarray(labels)
    split_mask = np.asarray(split_mask, dtype=bool)
    if labels.shape != features.valid_mask.shape:
        raise ValueError(
            "labels and feature valid_mask must have identical shapes: "
            f"{labels.shape} != {features.valid_mask.shape}"
        )
    if split_mask.shape != labels.shape:
        raise ValueError(
            f"split_mask shape {split_mask.shape} does not match labels {labels.shape}"
        )

    selected = features.valid_mask & split_mask
    flat_indices = np.flatnonzero(selected)
    if flat_indices.size == 0:
        return (
            np.empty((0, len(features.feature_names)), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    return (
        features.matrix[_valid_flat_positions(features.valid_mask, flat_indices)],
        labels.ravel()[flat_indices].astype(np.uint8, copy=False),
    )


def _valid_flat_positions(valid_mask: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    """Map raster flat indices to row positions in FeatureResult.matrix."""
    valid_flat = np.flatnonzero(valid_mask)
    return np.searchsorted(valid_flat, flat_indices)


def cap_samples_per_class(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_per_class: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap samples per class without changing class IDs."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D and y must be 1-D with matching row counts")
    if max_per_class is None:
        return X, y
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in np.unique(y):
        indices = np.flatnonzero(y == class_id)
        if indices.size > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        selected.append(indices)

    if not selected:
        return X[:0], y[:0]
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return X[indices], y[indices]


def balanced_class_weights(y: np.ndarray, num_classes: int = N_CLASSES) -> dict[int, float]:
    """Return inverse-frequency class weights for classes present in y."""
    y = np.asarray(y)
    counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    present = counts > 0
    if not np.any(present):
        raise ValueError("Cannot compute class weights from an empty label array")
    total = float(y.size)
    present_count = int(np.sum(present))
    return {
        class_id: total / (present_count * float(counts[class_id]))
        for class_id in range(num_classes)
        if counts[class_id] > 0
    }


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int = N_CLASSES,
    random_state: int = RANDOM_SEED,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    **kwargs: Any,
):
    """Train a conservative LightGBM multiclass model.

    LightGBM is imported lazily so the GEOCORE runtime does not require it.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required for offline ML training. "
            "Install dependencies from requirements-ml.txt."
        ) from exc

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D and y must be 1-D with matching row counts")
    if X.shape[0] == 0:
        raise ValueError("Cannot train LightGBM on an empty dataset")
    if np.any((y < 0) | (y >= num_classes)):
        raise ValueError(f"y contains class IDs outside 0..{num_classes - 1}")

    weights = balanced_class_weights(y, num_classes=num_classes)
    sample_weight = np.asarray([weights[int(label)] for label in y], dtype=np.float32)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        verbosity=-1,
        **kwargs,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Calculate deterministic multiclass metrics without sklearn."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")

    n_classes = len(class_names)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        if 0 <= truth < n_classes and 0 <= prediction < n_classes:
            matrix[truth, prediction] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(class_names):
        tp = float(matrix[class_id, class_id])
        support = float(np.sum(matrix[class_id, :]))
        predicted = float(np.sum(matrix[:, class_id]))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    baseline_predictions: np.ndarray | None = None,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Evaluate a fitted model and optionally compare a baseline on the same test set."""
    predictions = np.asarray(model.predict(np.asarray(X, dtype=np.float32)))
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    predictions = predictions.astype(np.int64, copy=False)

    result = evaluate_predictions(y, predictions, class_names=class_names)
    if baseline_predictions is not None:
        baseline = evaluate_predictions(
            y, np.asarray(baseline_predictions), class_names=class_names
        )
        result["baseline"] = baseline
    return result


def save_model_artifact(
    model: Any,
    output_dir: str | Path,
    *,
    feature_names: tuple[str, ...],
    reflectance_scaling: dict[str, dict[str, float]] | None,
    label_source: str,
    aois: list[str],
    dates: list[str],
    metrics: dict[str, Any] | None = None,
    version: str = "0.1.0",
) -> dict[str, str]:
    """Save a LightGBM model and an auditable experimental model card."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_path = output / "land_cover_lightgbm.txt"
    model.save_model(str(model_path))

    card = {
        "status": "EXPERIMENTAL",
        "version": version,
        "feature_names": list(feature_names),
        "class_names": list(CLASS_NAMES),
        "reflectance_scaling": reflectance_scaling,
        "label_source": label_source,
        "aois": list(aois),
        "dates": list(dates),
        "metrics": metrics if metrics is not None else "NOT_EVALUATED",
    }
    card_path = output / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return {"model": str(model_path), "model_card": str(card_path)}
