from __future__ import annotations

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

import rainbear
from rainbear import ZarrBackend


def test_mem() -> None:
    """Opt-in memory/regression test on a large external dataset.

    This test is intentionally skipped by default because it depends on a local dataset
    and can be very memory-intensive.
    """
    dataset = os.environ.get("RAINBEAR_MEM_DATASET")
    if not dataset:
        pytest.skip(
            "Set RAINBEAR_MEM_DATASET to run this test against a local large Zarr store."
        )

    scz = rainbear.scan_zarr(dataset)
    scz = scz.filter(
        (pl.col("time") == datetime(2024, 1, 1, 0, 0, 0))
        &
        # lead_time now recognized as Duration (units: hours)
        (pl.col("lead_time") == timedelta(hours=1))
        & (pl.col("y") >= 5)
        & (pl.col("y") <= 10)
        & (pl.col("x") >= 5)
        & (pl.col("x") <= 15)
    ).select(["time", "latitude", "longitude", "2m_temperature"])
    print("reached here 1")
    df = scz.collect()
    assert df.height == 66
    print("reached here 2")
    print(df)
    print("reached here 3")


def test_remote_mem() -> None:
    """Test a remote dataset"""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    scz = rainbear.scan_zarr(remote_ds, max_chunks_to_read=10)
    # Note: x coordinate array starts at 7, y starts at 11 (not 0)
    # The filter x==7, y==11 selects the first spatial point
    # With time and lead_time filters, we expect 1 row (single point in space-time)
    scz = scz.filter(
        (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0))
        & (pl.col("lead_time") == timedelta(hours=1))
        & (pl.col("y") == 11)  # First y coordinate value
        & (pl.col("x") == 7)  # First x coordinate value
    ).select(["time", "lead_time", "80m_wind_speed"])
    df = scz.collect()
    assert df.columns == ["time", "lead_time", "80m_wind_speed"]
    print(df)
    assert df.height == 1


def test_remote_mem3() -> None:
    """Test a remote dataset"""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    scz = rainbear.scan_zarr(remote_ds, max_chunks_to_read=30)
    scz = scz.filter(
        (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0))
        & (pl.col("lead_time") == timedelta(hours=1))
        & (pl.col("y") == 12)
        & (pl.col("x") == 12)
        # (pl.col("y") <= 6) &
        # (pl.col("x") >= 5) &
        # (pl.col("x") <= 6)
    ).select(["time", "lead_time", "x", "y"])
    df = scz.collect()
    assert df.columns == ["time", "lead_time", "x", "y"]
    print(df)
    assert df.height == 1


@pytest.mark.asyncio
async def test_remote_mem4() -> None:
    """Test a remote dataset"""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    df = await rainbear.scan_zarr_async(
        remote_ds,
        (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0))
        & (pl.col("lead_time") == timedelta(hours=1))
        & (pl.col("y") == 13)
        & (pl.col("x") == 13),
        variables=["80m_wind_speed"],
        with_columns=["time", "lead_time", "x", "y", "80m_wind_speed"]
    )
    # Column order depends on dataset dimension order (y comes before x in this dataset)
    assert df.columns == ["time", "lead_time", "y", "x", "80m_wind_speed"]
    print(df)
    assert df.height == 1


def test_remote_mem2() -> None:
    """Test a remote dataset"""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    [ch, coord_reads] = ZarrBackend.from_url(remote_ds).selected_chunks_debug(
        (
            (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0))
            & (pl.col("lead_time") == timedelta(hours=1))
            & (pl.col("y") == 59)
            & (pl.col("x") == 100)
            # (pl.col("y") <= 6) &
            # (pl.col("x") >= 5) &
            # (pl.col("x") <= 6)
        ),
    )
    print(ch)
    assert coord_reads == 5
    assert ch == [
        {
            "indices": [17496, 0, 0, 0],
            "origin": [17496, 0, 0, 0],
            "shape": [1, 31, 54, 108],
        }
    ]


if __name__ == "__main__":
    test_remote_mem()
