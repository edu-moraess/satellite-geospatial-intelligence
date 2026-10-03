import unittest
from tempfile import TemporaryDirectory

from src.ml.dataset_manifest import (
    BIOME_DOMAINS,
    CLIMATE_DOMAINS,
    WORLD_COVER_CLASSES,
    DatasetSample,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def _sample(sample_id="sample-001"):
    return DatasetSample(
        sample_id=sample_id,
        source="ESA WorldCover",
        scene_id="S2A_TEST",
        latitude=-15.78,
        longitude=-47.93,
        country="Brazil",
        continent="South America",
        biome="Cerrado",
        climate_domain="Tropical",
        worldcover_class=40,
        acquisition_date="2021-11-25T13:20:00Z",
        spatial_block="22LHH:0001:0002",
        geocore_class=1,
    )


class TestDatasetManifest(unittest.TestCase):
    def test_worldcover_registry_preserves_all_11_source_classes(self):
        self.assertEqual(
            tuple(WORLD_COVER_CLASSES),
            (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100),
        )

    def test_domain_registries_are_explicit(self):
        self.assertIn("Cerrado", BIOME_DOMAINS)
        self.assertIn("Amazon", BIOME_DOMAINS)
        self.assertIn("Semi-arid", CLIMATE_DOMAINS)
        self.assertIn("Temperate", CLIMATE_DOMAINS)

    def test_sample_round_trips_through_jsonl_manifest(self):
        records = [_sample("b"), _sample("a")]
        with TemporaryDirectory() as tmp:
            path = write_manifest(records, f"{tmp}/samples.jsonl")
            loaded = read_manifest(path)

        self.assertEqual([item.sample_id for item in loaded], ["a", "b"])
        self.assertEqual(loaded[0].worldcover_class, 40)
        self.assertEqual(loaded[0].geocore_class, 1)

    def test_duplicate_sample_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_manifest([_sample("same"), _sample("same")])

    def test_invalid_worldcover_code_is_rejected(self):
        with self.assertRaises(ValueError):
            _sample().__class__(
                **{**_sample().__dict__, "worldcover_class": 999}
            )

    def test_invalid_spatial_block_is_rejected(self):
        with self.assertRaises(ValueError):
            _sample().__class__(
                **{**_sample().__dict__, "spatial_block": ""}
            )


if __name__ == "__main__":
    unittest.main()
