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
        bbox = create_bbox(latitude=-15.0, longitude=-47.0, area_size=0.2)
        min_lon, min_lat, max_lon, max_lat = bbox
        self.assertAlmostEqual((min_lon + max_lon) / 2, -47.0)
        self.assertAlmostEqual((min_lat + max_lat) / 2, -15.0)

    def test_bbox_ordering(self):
        bbox = create_bbox(latitude=-15.0, longitude=-47.0, area_size=0.2)
        min_lon, min_lat, max_lon, max_lat = bbox
        self.assertLess(min_lon, max_lon)
        self.assertLess(min_lat, max_lat)

    def test_larger_area_produces_larger_bbox(self):
        small = create_bbox(latitude=0.0, longitude=0.0, area_size=0.1)
        large = create_bbox(latitude=0.0, longitude=0.0, area_size=0.5)
        self.assertLess(large[0], small[0])
        self.assertGreater(large[2], small[2])


class TestAoiGeometry(unittest.TestCase):

    def _square_feature(self):
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-48.0, -16.0],
                        [-46.0, -16.0],
                        [-46.0, -14.0],
                        [-48.0, -14.0],
                        [-48.0, -16.0],
                    ]
                ],
            },
            "properties": {},
        }

    def test_geometry_type_polygon(self):
        feature = self._square_feature()
        self.assertEqual(geometry_type(feature), "Polygon")

    def test_geometry_bbox_matches_known_rectangle(self):
        feature = self._square_feature()
        bbox = geometry_bbox(feature)
        self.assertEqual(bbox, [-48.0, -16.0, -46.0, -14.0])

    def test_bbox_center(self):
        center = bbox_center([-48.0, -16.0, -46.0, -14.0])
        self.assertAlmostEqual(center[0], -47.0)
        self.assertAlmostEqual(center[1], -15.0)

    def test_bbox_area_km2_is_positive(self):
        area = bbox_area_km2([-48.0, -16.0, -46.0, -14.0])
        self.assertGreater(area, 0.0)

    def test_build_aoi_from_feature(self):
        feature = self._square_feature()
        aoi = build_aoi(feature)
        self.assertIn("bbox", aoi)
        self.assertIn("geometry", aoi)

    def test_format_bbox_is_human_readable(self):
        text = format_bbox([-48.0, -16.0, -46.0, -14.0])
        self.assertIsInstance(text, str)
        self.assertIn("-48", text)

    def test_get_selected_aoi_none_when_map_state_is_none(self):
        self.assertIsNone(get_selected_aoi(None))

    def test_get_selected_aoi_none_when_no_drawings(self):
        self.assertIsNone(get_selected_aoi({"all_drawings": []}))

    def test_get_selected_aoi_returns_last_drawing(self):
        feature = self._square_feature()
        state = {"all_drawings": [feature]}
        aoi = get_selected_aoi(state)
        self.assertIsNotNone(aoi)


if __name__ == "__main__":
    unittest.main()
