"""Basic scan_zarr functionality tests."""

from __future__ import annotations

import polars as pl
import pytest

import rainbear


def test_scan_zarr_smoke(baseline_datasets: dict[str, str]) -> None:
    zarr_url = baseline_datasets["orography_chunked_10x10"]
    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    assert df.height == 16 * 20
    # Dataset has y, x dimensions plus 3 data variables
    assert df.columns == ["y", "x", "geopotential_height", "latitude", "longitude"]


def test_sel_predicate(baseline_datasets: dict[str, str]) -> None:
    zarr_url = baseline_datasets["orography_chunked_10x10"]
    lf = rainbear.scan_zarr(zarr_url)
    lf = lf.filter(
        (pl.col("y") >= 3)
        & (pl.col("y") <= 10)
        & (pl.col("x") >= 4)
        & (pl.col("x") <= 12)
    )
    df = lf.collect()

    assert df.height == (10 - 3 + 1) * (12 - 4 + 1)
    assert df.filter(pl.col("y") < 3).is_empty()
    assert df.filter(pl.col("y") > 10).is_empty()
    assert df.filter(pl.col("x") < 4).is_empty()
    assert df.filter(pl.col("x") > 12).is_empty()


def test_max_chunks_to_read(baseline_datasets: dict[str, str]) -> None:
    zarr_url = baseline_datasets["orography_chunked_10x10"]

    lf = rainbear.scan_zarr(
        zarr_url, variables=["geopotential_height"], max_chunks_to_read=1
    )

    with pytest.raises(pl.exceptions.ComputeError, match="max_chunks_to_read"):
        lf.collect()
