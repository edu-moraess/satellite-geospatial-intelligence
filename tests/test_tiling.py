"""
Tests: src.tiling
"""

import unittest

import numpy as np

from src.tiling import create_tiles, tile_count


class TestTiling(unittest.TestCase):

    def test_tile_size_validation(self):
        image = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaises(ValueError):
            create_tiles(image, tile_size=0)

    def test_overlap_must_be_smaller_than_tile_size(self):
        image = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaises(ValueError):
            create_tiles(image, tile_size=64, overlap=64)

    def test_none_image_raises(self):
        with self.assertRaises(ValueError):
            create_tiles(None, tile_size=64)

    def test_tiles_have_consistent_size(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)

        tiles = create_tiles(image, tile_size=128, overlap=16)

        for tile in tiles:
            self.assertEqual(
                tile["image"].shape[:2], (128, 128)
            )

    def test_border_tiles_are_covered(self):
        # 300 is not a clean multiple of (128 - 16), so the
        # bottom/right border must still be fully covered by
        # an extra tile aligned to the image edge.
        image = np.zeros((300, 300), dtype=np.uint8)

        tiles = create_tiles(image, tile_size=128, overlap=16)

        max_y = max(t["y"] for t in tiles)
        max_x = max(t["x"] for t in tiles)

        self.assertEqual(max_y + 128, 300)
        self.assertEqual(max_x + 128, 300)

    def test_small_image_still_returns_one_tile(self):
        # Image smaller than tile_size must still produce
        # exactly one (padded) tile, not zero.
        image = np.zeros((50, 50), dtype=np.uint8)

        tiles = create_tiles(image, tile_size=128, overlap=16)

        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0]["image"].shape, (128, 128))
        self.assertEqual(tiles[0]["width"], 50)
        self.assertEqual(tiles[0]["height"], 50)

    def test_tile_count_matches_create_tiles(self):
        image = np.zeros((256, 256), dtype=np.uint8)

        count = tile_count(image, tile_size=128, overlap=32)
        tiles = create_tiles(image, tile_size=128, overlap=32)

        self.assertEqual(count, len(tiles))


if __name__ == "__main__":
    unittest.main()
