"""
Tests: src.catalog (bbox) and src.aoi (Area of Interest).
"""

import unittest

from src.catalog import create_bbox
from src.aoi import (
    geometry_type,
    geometry_bbox,
    bbox_center,
    bbox_area_km2,
    build_aoi,
    get_selected_aoi,
    format_bbox,
)


class TestCreateBbox(unittest.TestCase):

    def test_bbox_is_centered_on_point(self):
        min_lon, min_lat, max_lon, max_lat = create_bbox(
            latitude=0.0, longitude=0.0, area_size=10.0
        )

        # Symmetric around the origin.
        self.assertAlmostEqual(min_lon, -max_lon, places=6)
        self.assertAlmostEqual(min_lat, -max_lat, places=6)

    def test_bbox_ordering(self):
        min_lon, min_lat, max_lon, max_lat = create_bbox(
            latitude=-23.5, longitude=-46.6, area_size=5.0
        )

        self.assertLess(min_lon, max_lon)
        self.assertLess(min_lat, max_lat)

    def test_larger_area_produces_larger_bbox(self):
        small = create_bbox(latitude=10, longitude=10, area_size=2.0)
        large = create_bbox(latitude=10, longitude=10, area_size=20.0)

        small_width = small[2] - small[0]
        large_width = large[2] - large[0]

        self.assertGreater(large_width, small_width)


class TestAoiGeometry(unittest.TestCase):

    def _rectangle_feature(self):
        # A simple ~1x1 degree rectangle near the equator.
        return {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [10.0, 10.0],
                    [11.0, 10.0],
                    [11.0, 11.0],
                    [10.0, 11.0],
                    [10.0, 10.0],
                ]],
            },
        }

    def test_geometry_type_polygon(self):
        feature = self._rectangle_feature()
        self.assertEqual(
            geometry_type(feature), "Polygon"
        )

    def test_geometry_bbox_matches_known_rectangle(self):
        feature = self._rectangle_feature()
        bbox = geometry_bbox(feature)

        self.assertAlmostEqual(bbox[0], 10.0, places=6)
        self.assertAlmostEqual(bbox[1], 10.0, places=6)
        self.assertAlmostEqual(bbox[2], 11.0, places=6)
        self.assertAlmostEqual(bbox[3], 11.0, places=6)

    def test_bbox_center(self):
        bbox = (10.0, 10.0, 11.0, 11.0)
        center_lon, center_lat = bbox_center(bbox)

        self.assertAlmostEqual(center_lon, 10.5, places=6)
        self.assertAlmostEqual(center_lat, 10.5, places=6)

    def test_bbox_area_km2_is_positive(self):
        bbox = (10.0, 10.0, 11.0, 11.0)
        area = bbox_area_km2(bbox)

        self.assertGreater(area, 0)

    def test_build_aoi_from_feature(self):
        feature = self._rectangle_feature()
        aoi = build_aoi(feature)

        self.assertEqual(aoi["geometry_type"], "Polygon")
        self.assertEqual(len(aoi["bbox"]), 4)
        self.assertGreater(aoi["area_km2"], 0)

    def test_get_selected_aoi_none_when_no_drawings(self):
        # No drawings at all -> AOI must be None, never a
        # fabricated/default geometry.
        map_state = {"all_drawings": []}
        self.assertIsNone(get_selected_aoi(map_state))

    def test_get_selected_aoi_none_when_map_state_is_none(self):
        self.assertIsNone(get_selected_aoi(None))

    def test_get_selected_aoi_returns_last_drawing(self):
        map_state = {
            "all_drawings": [
                self._rectangle_feature(),
            ]
        }

        aoi = get_selected_aoi(map_state)

        self.assertIsNotNone(aoi)
        self.assertEqual(aoi["geometry_type"], "Polygon")

    def test_format_bbox_is_human_readable(self):
        text = format_bbox((10.123456, 10.1, 11.0, 11.999999))
        self.assertIsInstance(text, str)
        self.assertIn(",", text)


if __name__ == "__main__":
    unittest.main()
