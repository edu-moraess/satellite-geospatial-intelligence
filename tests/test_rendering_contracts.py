"""Contract tests for GEOCORE rendering and sensor/workflow band mappings."""

import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.change_visualization import create_change_figure
from src.index_visualization import create_index_figure
from src.land_cover import create_land_cover_figure
from src.downloader import download_required_bands
from src.downloader_landsat import POSSIBLE_NAMES
from src.sensor_registry import SENSORS

ROOT = Path(__file__).resolve().parents[1]


def _subscript_keys(source: str, variable: str):
    tree = ast.parse(source)
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == variable:
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                keys.add(index.value)
    return keys


def test_rendering_functions_return_matplotlib_figures():
    figures = [
        create_index_figure(np.array([[0.1, -0.2], [0.3, np.nan]], dtype=np.float32), "NDVI"),
        create_land_cover_figure(np.array([[0, 1], [2, 4]], dtype=np.int8)),
        create_change_figure(np.array([[-1, 0], [1, np.nan]], dtype=np.float32)),
    ]
    try:
        from matplotlib.figure import Figure
        assert all(isinstance(figure, Figure) for figure in figures)
    finally:
        for figure in figures:
            plt.close(figure)


def test_app_uses_pyplot_for_all_three_matplotlib_figures():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    pyplot_calls = []
    plotly_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "st":
                if node.func.attr == "pyplot":
                    pyplot_calls.append(ast.get_source_segment(source, node))
                if node.func.attr == "plotly_chart":
                    plotly_calls.append(node)
    assert len(pyplot_calls) == 3
    assert any("index_figure" in call for call in pyplot_calls)
    assert any("classification_fig" in call for call in pyplot_calls)
    assert any('change_result["figure"]' in call for call in pyplot_calls)
    assert not plotly_calls


def test_sentinel_workflow_bands_match_registry_and_downloader():
    imagery = (ROOT / "src/workflows/imagery.py").read_text(encoding="utf-8")
    change = (ROOT / "src/workflows/change.py").read_text(encoding="utf-8")
    imagery_bands = _subscript_keys(imagery, "bands")
    change_bands = _subscript_keys(change, "before_bands") | _subscript_keys(change, "after_bands")
    expected = {"B02", "B03", "B04", "B08", "B11"}
    assert expected.issuperset(imagery_bands)
    assert expected.issuperset(change_bands)
    assert expected <= set(SENSORS["sentinel2"].bands.values())

    downloader_source = (ROOT / "src/downloader.py").read_text(encoding="utf-8")
    tree = ast.parse(downloader_source)
    required = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "required_bands":
                    required = {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}
    assert required == expected


def test_landsat_possible_names_use_the_same_sentinel_style_band_contract():
    assert set(POSSIBLE_NAMES) == {"B02", "B03", "B04", "B08", "B11"}


def test_download_required_bands_is_the_sentinel_download_contract():
    source = Path(download_required_bands.__code__.co_filename).read_text(encoding="utf-8")
    assert all(f'"{band}"' in source for band in ("B02", "B03", "B04", "B08", "B11"))
