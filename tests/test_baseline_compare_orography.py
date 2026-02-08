"""Tests comparing rainbear output against xarray baseline.

These tests verify that rainbear produces the same results as xarray for various
Zarr datasets and filter conditions. The key principle is:

1. Both rainbear and xarray scan the same Zarr store
2. The same filters are applied to both LazyFrames
3. Both are collected and compared

This tests the predicate pushdown functionality - filters are applied BEFORE
collection, not after.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

import rainbear

from .baseline_utils import assert_frames_equal, scan_via_xarray

# ---------------------------------------------------------------------------
# Test fixtures - generate test datasets once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def orography_path(tmp_path_factory) -> str:
    """Create a small orography dataset for testing."""
    from zarr.codecs import BloscCodec, BloscShuffle

    from . import zarr_generators
    
    path = tmp_path_factory.mktemp("data") / "orography.zarr"
    ds = zarr_generators.create_orography_dataset(nx=20, ny=16, sigma=4.0, seed=1)
    blosc = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
    ds.to_zarr(str(path), zarr_format=3, encoding={
        "geopotential_height": {"chunks": (8, 10), "compressors": [blosc]},
        "latitude": {"chunks": (8, 10), "compressors": [blosc]},
        "longitude": {"chunks": (8, 10), "compressors": [blosc]},
    })
    return str(path)


@pytest.fixture(scope="session")
def path(tmp_path_factory) -> str:
    """Create a small 4D dataset for testing."""
    from datetime import datetime, timedelta

    import numpy as np
    import xarray as xr
    from zarr.codecs import BloscCodec, BloscShuffle
    
    path = tmp_path_factory.mktemp("data") / "small.zarr"
    
    # Create a small synthetic dataset (much smaller than the full Grid)
    nx, ny, nt, nl = 20, 16, 2, 4
    x = np.arange(nx)
    y = np.arange(ny)
    time_values = [datetime(2024, 1, 1) + timedelta(hours=i * 6) for i in range(nt)]
    lead_time_values = [timedelta(hours=i) for i in range(nl)]
    
    temp_data = 273.15 + 10 * np.random.randn(nt, nl, ny, nx)
    
    ds = xr.Dataset(
        data_vars={
            "temperature": (["time", "lead_time", "y", "x"], temp_data),
        },
        coords={
            "time": time_values,
            "lead_time": lead_time_values,
            "x": x,
            "y": y,
        },
    )
    
    blosc = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
    ds.to_zarr(str(path), zarr_format=3, encoding={
        "temperature": {"chunks": (1, 2, 8, 10), "compressors": [blosc]},
    })
    return str(path)


# ---------------------------------------------------------------------------
# Basic scan tests - no filters
# ---------------------------------------------------------------------------

def test_scan_orography_no_filter(orography_path: str) -> None:
    """Test that scanning without filters matches xarray."""
    columns = ["y", "x", "geopotential_height"]
    
    out = rainbear.scan_zarr(orography_path).collect()
    baseline = scan_via_xarray(orography_path, columns=columns).collect()
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["y", "x"])


def test_scan_no_filter(path: str) -> None:
    """Test that scanning a 4D dataset without filters matches xarray."""
    columns = ["time", "lead_time", "y", "x", "temperature"]
    
    out = rainbear.scan_zarr(path).collect()
    baseline = scan_via_xarray(path, columns=columns).collect()
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["time", "lead_time", "y", "x"])


# ---------------------------------------------------------------------------
# Filter tests on integer dimensions (y, x)
# ---------------------------------------------------------------------------

def test_filter_on_y_dimension(orography_path: str) -> None:
    """Test filtering on y dimension."""
    columns = ["y", "x", "geopotential_height"]
    filter_expr = (pl.col("y") >= 5) & (pl.col("y") <= 10)
    
    out = (
        rainbear.scan_zarr(orography_path)
        .filter(filter_expr)
        .collect()
    )
    baseline = (
        scan_via_xarray(orography_path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["y", "x"])


def test_filter_on_x_dimension(orography_path: str) -> None:
    """Test filtering on x dimension."""
    columns = ["y", "x", "geopotential_height"]
    filter_expr = (pl.col("x") >= 3) & (pl.col("x") < 15)
    
    out = (
        rainbear.scan_zarr(orography_path)
        .filter(filter_expr)
        .collect()
    )
    baseline = (
        scan_via_xarray(orography_path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["y", "x"])


def test_filter_on_both_dimensions(orography_path: str) -> None:
    """Test filtering on both y and x dimensions."""
    columns = ["y", "x", "geopotential_height"]
    filter_expr = (pl.col("y") >= 4) & (pl.col("y") <= 12) & (pl.col("x") >= 5) & (pl.col("x") <= 15)
    
    out = (
        rainbear.scan_zarr(orography_path)
        .filter(filter_expr)
        .collect()
    )
    baseline = (
        scan_via_xarray(orography_path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["y", "x"])


# ---------------------------------------------------------------------------
# Filter tests on float coordinates (latitude, longitude)
# ---------------------------------------------------------------------------

def test_filter_on_latitude(orography_path: str) -> None:
    """Test filtering on latitude (2D float coordinate)."""
    columns = ["y", "x", "geopotential_height", "latitude"]
    filter_expr = ((pl.col("latitude") >= 20.05) & (pl.col("latitude") <= 20.15))
    
    out = rainbear.scan_zarr(orography_path).filter(filter_expr).select(columns).collect()
    baseline = (
        scan_via_xarray(orography_path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["y", "x"])


# ---------------------------------------------------------------------------
# Filter tests on datetime dimensions
# ---------------------------------------------------------------------------

def test_filter_on_time_equality(path: str) -> None:
    """Test filtering on time dimension with equality."""
    from datetime import datetime
    
    columns = ["time", "lead_time", "y", "x", "temperature"]
    target_time = datetime(2024, 1, 1, 0, 0, 0)
    filter_expr = (pl.col("time") == target_time)
    
    out = (
        rainbear.scan_zarr(path)
        .filter(filter_expr)
        .select(columns)
        .collect()
    )
    baseline = (
        scan_via_xarray(path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["time", "lead_time", "y", "x"])


def test_filter_on_lead_time(path: str) -> None:
    """Test filtering on lead_time (duration) dimension."""
    columns = ["time", "lead_time", "y", "x", "temperature"]
    # Filter for lead_time between 1 and 2 hours
    filter_expr = ((pl.col("lead_time") >= timedelta(hours=1)) & (pl.col("lead_time") <= timedelta(hours=2)))
    
    out = (
        rainbear.scan_zarr(path)
        .filter(filter_expr)
        .select(columns)
        .collect()
    )
    baseline = (
        scan_via_xarray(path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["time", "lead_time", "y", "x"])


def test_combined_time_and_spatial_filter(path: str) -> None:
    """Test combining time and spatial filters."""
    from datetime import datetime
    
    columns = ["time", "lead_time", "y", "x", "temperature"]
    filter_expr = (
        (pl.col("time") == datetime(2024, 1, 1, 0, 0, 0)) &
        (pl.col("y") >= 5) & 
        (pl.col("y") <= 10) &
        (pl.col("x") >= 5) &
        (pl.col("x") <= 15)
    )
    
    out = (
        rainbear.scan_zarr(path)
        .filter(filter_expr)
        .select(columns)
        .collect()
    )
    baseline = (
        scan_via_xarray(path, columns=columns)
        .filter(filter_expr)
        .collect()
    )
    
    assert_frames_equal(out.select(columns), baseline, sort_by=["time", "lead_time", "y", "x"])


# ---------------------------------------------------------------------------
# Schema and dtype tests
# ---------------------------------------------------------------------------

def test_datetime_dtype_matches(path: str) -> None:
    """Test that datetime columns have correct dtype."""
    out = rainbear.scan_zarr(path).collect()
    
    assert out.schema["time"] == pl.Datetime("ns")
    assert out.schema["lead_time"] == pl.Duration("ns")


def test_schema_matches_xarray(path: str) -> None:
    """Test that schema matches xarray baseline."""
    columns = ["time", "lead_time", "y", "x", "temperature"]
    
    out = rainbear.scan_zarr(path).collect()
    baseline = scan_via_xarray(path, columns=columns).collect()
    
    # Compare schemas
    for col in columns:
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
        assert out.schema[col] == baseline.schema[col], f"Schema mismatch for {col}"
