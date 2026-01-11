"""Basic scan_zarr functionality tests."""

from __future__ import annotations

import polars as pl

import rainbear
from rainbear import _core


def test_scan_zarr_smoke(dataset_path) -> None:
    path = dataset_path("demo_store.zarr")
    _core._create_demo_store(path)

    lf = rainbear.scan_zarr(path)
    df = lf.collect()

    assert df.height == 12
    assert df.columns == ["time", "lat", "temp"]


def test_sel_predicate(dataset_path) -> None:
    path = dataset_path("demo_store_sel.zarr")
    _core._create_demo_store(path)

    lf = rainbear.scan_zarr(path)
    lf = lf.filter((pl.col("lat") >= 20.0) & (pl.col("lat") <= 30.0))
    df = lf.collect()

    assert df.height == 8
    assert df["lat"].is_in([20.0, 30.0]).all()


def test_max_chunks_to_read(dataset_path) -> None:
    path = dataset_path("demo_store_max_chunks.zarr")
    _core._create_demo_store(path)

    # Demo store temp has 2 chunks (time chunked as [2, 2]), so limiting to 1 chunk must fail.
    lf = rainbear.scan_zarr(path, max_chunks_to_read=1)
    import pytest

    with pytest.raises(pl.exceptions.ComputeError, match="max_chunks_to_read"):
        lf.collect()
