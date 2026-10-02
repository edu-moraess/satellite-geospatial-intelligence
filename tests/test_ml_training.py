import unittest
from tempfile import TemporaryDirectory

import numpy as np

from src.ml.features import FeatureResult
from src.ml.training import (
    balanced_class_weights,
    cap_samples_per_class,
    evaluate_predictions,
    prepare_split,
    save_model_artifact,
    spatial_block_split,
)


class FakeModel:
    def __init__(self):
        self.saved_to = None

    def save_model(self, path):
        self.saved_to = path


class TestMLTrainingUtilities(unittest.TestCase):
    def test_spatial_blocks_do_not_overlap(self):
        masks = spatial_block_split(
            (16, 16),
            block_size=4,
            train_fraction=0.5,
            validation_fraction=0.25,
            seed=7,
        )
        self.assertTrue(
            np.all(
                (masks["train"].astype(int)
                 + masks["validation"].astype(int)
                 + masks["test"].astype(int))
                == 1
            )
        )
        self.assertFalse(np.any(masks["train"] & masks["validation"]))
        self.assertFalse(np.any(masks["train"] & masks["test"]))
        self.assertFalse(np.any(masks["validation"] & masks["test"]))

    def test_prepare_split_uses_valid_feature_rows_only(self):
        valid_mask = np.array([[True, False], [True, True]])
        matrix = np.array(
            [[1, 2], [3, 4], [5, 6]],
            dtype=np.float32,
        )
        features = FeatureResult(
            matrix=matrix,
            feature_names=("f1", "f2"),
            valid_mask=valid_mask,
            reflectance_scaling=None,
        )
        labels = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        split = np.array([[True, True], [False, True]])
        X, y = prepare_split(features, labels, split)

        np.testing.assert_array_equal(X, np.array([[1, 2], [5, 6]], dtype=np.float32))
        np.testing.assert_array_equal(y, np.array([10, 40], dtype=np.uint8))

    def test_class_cap_is_deterministic(self):
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = np.array([0] * 5 + [1] * 5)
        X1, y1 = cap_samples_per_class(X, y, max_per_class=2, seed=11)
        X2, y2 = cap_samples_per_class(X, y, max_per_class=2, seed=11)

        self.assertEqual(X1.shape, (4, 2))
        self.assertEqual(y1.shape, (4,))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        self.assertEqual(np.bincount(y1).tolist(), [2, 2])

    def test_balanced_class_weights(self):
        weights = balanced_class_weights(np.array([0, 0, 0, 1, 1]))
        self.assertAlmostEqual(weights[0], 5.0 / 6.0)
        self.assertAlmostEqual(weights[1], 5.0 / 4.0)

    def test_metrics_include_per_class_and_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 2, 2])
        metrics = evaluate_predictions(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertEqual(len(metrics["confusion_matrix"]), 5)
        self.assertEqual(metrics["confusion_matrix"][0][0], 1)
        self.assertIn("Vegetation", metrics["per_class"])
        self.assertIn("f1", metrics["per_class"]["Water"])

    def test_model_card_is_experimental_and_preserves_contract(self):
        model = FakeModel()
        with TemporaryDirectory() as tmp:
            paths = save_model_artifact(
                model,
                tmp,
                feature_names=("B02", "NDVI"),
                reflectance_scaling=None,
                label_source="ESA WorldCover",
                aois=["test-aoi"],
                dates=["2026-01-01"],
            )
            from pathlib import Path
            import json

            card = json.loads(Path(paths["model_card"]).read_text(encoding="utf-8"))

        self.assertEqual(card["status"], "EXPERIMENTAL")
        self.assertEqual(card["feature_names"], ["B02", "NDVI"])
        self.assertEqual(card["metrics"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
