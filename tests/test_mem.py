from __future__ import annotations

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

import rainbear


def test_mem() -> None:
    """Opt-in memory/regression test on a large external dataset.

    This test is intentionally skipped by default because it depends on a local dataset
    and can be very memory-intensive.
    """
    dataset = os.environ.get("RAINBEAR_MEM_DATASET")
    if not dataset:
        pytest.skip("Set RAINBEAR_MEM_DATASET to run this test against a local large Zarr store.")

    scz = rainbear.scan_zarr(dataset)
    scz = scz.filter(
        (pl.col("time") == datetime(2024, 1, 1, 0, 0, 0)) &
        # lead_time now recognized as Duration (units: hours)
        (pl.col("lead_time") == timedelta(hours=1)) &
        (pl.col("y") >= 5) & 
        (pl.col("y") <= 10) &
        (pl.col("x") >= 5) &
        (pl.col("x") <= 15)
    ).select(
        ["time", "latitude", "longitude", "2m_temperature"]
    )
    print("reached here 1")
    df = scz.collect()
    assert df.height == 66
    print("reached here 2")
    print(df)
    print("reached here 3")


def test_remote_mem() -> None:
    """Test a remote dataset"""
    if os.environ.get("RAINBEAR_REMOTE_MEM") != "1":
        pytest.skip("Set RAINBEAR_REMOTE_MEM=1 to run this test against a remote dataset.")
    remote_ds = "gs://silurian-beta/forecasts/gft-us-v1_1.zarr"
    scz = rainbear.scan_zarr(remote_ds, max_chunks_to_read=10)
    scz = scz.filter(
        (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0)) &
        (pl.col("lead_time") == timedelta(hours=1)) &
        (pl.col("y") == 0) & 
        (pl.col("x") == 0)
        # (pl.col("y") <= 6) &
        # (pl.col("x") >= 5) &
        # (pl.col("x") <= 6)
    ).select(
        ["time", "lead_time", "80m_wind_speed"]
    )
    df = scz.collect()
    assert df.columns == ["time", "lead_time", "80m_wind_speed"]
    print(df)
    assert df.height == 24

if __name__ == "__main__":
    test_remote_mem()