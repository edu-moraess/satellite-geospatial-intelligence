"""
Tests: src.map_view (resolve_slider_default)

This is the guardrail behind the "ValueError: 650 is not in
iterable" bug. Requires streamlit/folium to import the module
(same as the rest of the app), so this runs wherever the app
itself runs.
"""

import unittest

from src.map_view import resolve_slider_default


class TestResolveSliderDefault(unittest.TestCase):

    OPTIONS = [500, 600, 650, 700, 800]

    def test_preferred_value_used_when_no_stored_value(self):
        result = resolve_slider_default(
            self.OPTIONS, stored_value=None, preferred_value=650
        )
        self.assertEqual(result, 650)

    def test_stored_value_takes_priority_when_valid(self):
        result = resolve_slider_default(
            self.OPTIONS, stored_value=700, preferred_value=650
        )
        self.assertEqual(result, 700)

    def test_invalid_stored_value_falls_back_to_preferred(self):
        # This is exactly the historical bug scenario: a
        # session_state value that is no longer a valid
        # option (e.g. left over from an older app version).
        result = resolve_slider_default(
            self.OPTIONS, stored_value=999, preferred_value=650
        )
        self.assertEqual(result, 650)

    def test_invalid_stored_and_invalid_preferred_falls_back_to_middle(self):
        result = resolve_slider_default(
            self.OPTIONS, stored_value=999, preferred_value=999
        )
        self.assertIn(result, self.OPTIONS)

    def test_result_is_always_in_options(self):
        # Property-style check across many combinations - the
        # one invariant that must NEVER break again.
        candidates = [None, 1, 500, 650, 800, 999, -1]

        for stored in candidates:
            for preferred in candidates:
                result = resolve_slider_default(
                    self.OPTIONS,
                    stored_value=stored,
                    preferred_value=preferred,
                )
                self.assertIn(result, self.OPTIONS)

    def test_empty_options_raises(self):
        with self.assertRaises(ValueError):
            resolve_slider_default([], stored_value=1, preferred_value=1)


if __name__ == "__main__":
    unittest.main()
