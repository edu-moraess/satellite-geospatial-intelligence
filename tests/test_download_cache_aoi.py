"""Tests for AOI-aware raster download caching without network access."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import rasterio
from rasterio.transform import from_origin

from src import downloader, downloader_landsat


def _write_source(path: Path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(10, 10, 1, 1),
    ) as dst:
        dst.write(__import__("numpy").ones((1, 10, 10), dtype="uint16"))


def _item(source: Path):
    return SimpleNamespace(
        id="SCENE_TEST",
        assets={
            "blue": SimpleNamespace(href=str(source)),
            "SR_B2": SimpleNamespace(href=str(source)),
        },
    )


def test_bbox_cache_key_is_deterministic_and_rounds_float_noise():
    a = [11.0000001, 1.0, 12.0, 2.0]
    b = [11.0000002, 1.0, 12.0, 2.0]
    assert downloader.bbox_cache_key(a) == downloader.bbox_cache_key(b)
    assert downloader.bbox_cache_key(a) != downloader.bbox_cache_key([11.0, 1.0, 13.0, 2.0])


def test_sentinel_cache_reuses_same_bbox_but_not_different_bbox(tmp_path):
    source = tmp_path / "source.tif"
    _write_source(source)
    item = _item(source)
    aoi_a = [11.0, 1.0, 12.0, 2.0]
    aoi_b = [12.0, 1.0, 13.0, 2.0]
    real_open = downloader.rasterio.open
    remote_reads = 0

    def counted_open(path, *args, **kwargs):
        nonlocal remote_reads
        if str(path) == str(source):
            remote_reads += 1
        return real_open(path, *args, **kwargs)

    with mock.patch.object(downloader.rasterio, "open", side_effect=counted_open):
        first = downloader.download_band(item, "B02", aoi_a, tmp_path / "raw" / item.id)
        second = downloader.download_band(item, "B02", aoi_a, tmp_path / "raw" / item.id)
        third = downloader.download_band(item, "B02", aoi_b, tmp_path / "raw" / item.id)

    assert first == second
    assert first != third
    assert downloader.bbox_cache_key(aoi_a) in first.parts
    assert downloader.bbox_cache_key(aoi_b) in third.parts
    assert remote_reads == 2


def test_landsat_cache_reuses_same_bbox_but_not_different_bbox(tmp_path):
    source = tmp_path / "source.tif"
    _write_source(source)
    item = _item(source)
    aoi_a = [11.0, 1.0, 12.0, 2.0]
    aoi_b = [12.0, 1.0, 13.0, 2.0]
    real_open = downloader_landsat.rasterio.open
    remote_reads = 0

    def counted_open(path, *args, **kwargs):
        nonlocal remote_reads
        if str(path) == str(source):
            remote_reads += 1
        return real_open(path, *args, **kwargs)

    with mock.patch.object(downloader_landsat.rasterio, "open", side_effect=counted_open):
        first = downloader_landsat._download_windowed_band(item, "B02", item.assets["SR_B2"], aoi_a, tmp_path / "raw" / item.id)
        second = downloader_landsat._download_windowed_band(item, "B02", item.assets["SR_B2"], aoi_a, tmp_path / "raw" / item.id)
        third = downloader_landsat._download_windowed_band(item, "B02", item.assets["SR_B2"], aoi_b, tmp_path / "raw" / item.id)

    assert first == second
    assert first != third
    assert downloader.bbox_cache_key(aoi_a) in first.parts
    assert downloader.bbox_cache_key(aoi_b) in third.parts
    assert remote_reads == 2
