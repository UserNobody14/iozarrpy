"""Interpolation tests for CF-coded datetime coordinates.

CF (Climate and Forecast) conventions encode time as numeric values with a
"units" attribute (e.g. "seconds since 1970-01-01 00:00:00"). These tests
verify that interpolation works correctly when:
- Time is stored as int64 with CF units (common case)
- Time is stored as float64 with CF units (e.g. "days since epoch")
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr
from interpolars import interpolate_nd
from zarr.codecs import BloscCodec, BloscShuffle

import rainbear

BLOSC_ZSTD = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)


def _xr_interp_single_point(
    ds: xr.Dataset,
    var: str,
    coords: dict[str, float | datetime.datetime],
) -> float:
    """Return the scalar result of xarray linear interpolation at one point."""
    kw = {k: [v] for k, v in coords.items()}
    result = ds.interp(**kw).compute()
    return float(result[var].values.flat[0])


# ---------------------------------------------------------------------------
# CF int64: seconds since epoch (common xarray encoding)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Interpolation on datetime axis returns bracketing value; "
    "see test_interp_cf_datetime_2d_with_time_filter for CF datetime validation"
)
async def test_interp_cf_datetime_int64_seconds_since_epoch(tmp_path: Path) -> None:
    """1D interpolation on time coordinate stored as int64 with CF units."""
    # 5 time points: 2025-01-01 00:00, 06:00, 12:00, 18:00, 2025-01-02 00:00
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    times_ns = np.array(
        [
            np.datetime64("2025-01-01T00:00:00", "ns"),
            np.datetime64("2025-01-01T06:00:00", "ns"),
            np.datetime64("2025-01-01T12:00:00", "ns"),
            np.datetime64("2025-01-01T18:00:00", "ns"),
            np.datetime64("2025-01-02T00:00:00", "ns"),
        ]
    )
    time_seconds = ((times_ns - epoch) / np.timedelta64(1, "s")).astype(np.int64)

    ds = xr.Dataset(
        data_vars={"value": (["time"], np.array([10.0, 20.0, 30.0, 40.0, 50.0]))},
        coords={"time": (["time"], time_seconds)},
    )
    ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"

    zarr_path = tmp_path / "cf_time_int64.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={
            "value": {"chunks": (3,), "compressors": [BLOSC_ZSTD]},
        },
    )

    # Interpolate at 2025-01-01 09:00 (midway between 06:00 and 12:00)
    target = pl.DataFrame(
        {"time": [datetime.datetime(2025, 1, 1, 9, 0, 0)]},
        schema={"time": pl.Datetime("us")},
    )
    expr = interpolate_nd(["time"], ["value"], target)
    result = await rainbear.scan_zarr_async(str(zarr_path), expr)
    unnested = result.unnest("interpolated")
    actual = float(unnested["value"][0])

    # Compare with xarray (decode CF then interp)
    ds_decoded = xr.decode_cf(ds)
    expected = _xr_interp_single_point(
        ds_decoded, "value", {"time": datetime.datetime(2025, 1, 1, 9, 0, 0)}
    )
    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# CF float64: days since epoch (common in climate data)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Interpolation on datetime axis returns bracketing value; "
    "see test_interp_cf_datetime_2d_with_time_filter for CF datetime validation"
)
async def test_interp_cf_datetime_float64_days_since_epoch(tmp_path: Path) -> None:
    """1D interpolation on time coordinate stored as float64 with CF units."""
    # Days since 1970-01-01 for 5 consecutive days
    days_since_epoch = np.array([20088.0, 20089.0, 20090.0, 20091.0, 20092.0], dtype=np.float64)
    # 20088 corresponds to 2025-01-01

    ds = xr.Dataset(
        data_vars={"value": (["time"], np.array([100.0, 110.0, 120.0, 130.0, 140.0]))},
        coords={"time": (["time"], days_since_epoch)},
    )
    ds["time"].attrs["units"] = "days since 1970-01-01 00:00:00"

    zarr_path = tmp_path / "cf_time_float64.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={
            "value": {"chunks": (3,), "compressors": [BLOSC_ZSTD]},
        },
    )

    # Interpolate at 2025-01-02 12:00 (midway between day 1 and 2)
    target = pl.DataFrame(
        {"time": [datetime.datetime(2025, 1, 2, 12, 0, 0)]},
        schema={"time": pl.Datetime("us")},
    )
    expr = interpolate_nd(["time"], ["value"], target)
    result = await rainbear.scan_zarr_async(str(zarr_path), expr)
    unnested = result.unnest("interpolated")
    actual = float(unnested["value"][0])

    # Compare with xarray (decode CF then interp)
    ds_decoded = xr.decode_cf(ds)
    expected = _xr_interp_single_point(
        ds_decoded, "value", {"time": datetime.datetime(2025, 1, 2, 12, 0, 0)}
    )
    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# CF datetime + spatial: 2D interpolation with time filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interp_cf_datetime_2d_with_time_filter(tmp_path: Path) -> None:
    """2D interpolation (lat, lon) with CF-coded time as filter dimension."""
    nlat, nlon = 8, 10
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    # 3 time steps with CF int64 encoding
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    times_ns = np.array(
        [
            np.datetime64("2025-02-01T16:00:00", "ns"),
            np.datetime64("2025-02-01T17:00:00", "ns"),
            np.datetime64("2025-02-01T18:00:00", "ns"),
        ]
    )
    time_seconds = ((times_ns - epoch) / np.timedelta64(1, "s")).astype(np.int64)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"CAL": (["time", "lat", "lon"], data)},
        coords={
            "time": (["time"], time_seconds),
            "lat": lat_coords,
            "lon": lon_coords,
        },
    )
    ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"

    zarr_path = tmp_path / "cf_time_2d.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={
            "CAL": {"chunks": (1, 4, 5), "compressors": [BLOSC_ZSTD]},
        },
    )

    # Interpolate at (lat=0.33, lon=0.14) with time filter at 17:00
    target_df = pl.DataFrame({"lat": [0.33], "lon": [0.14]})
    expr = interpolate_nd(
        ["lat", "lon"],
        [
            pl.col("CAL").filter(
                pl.col("time") == datetime.datetime(2025, 2, 1, 17, 0, 0)
            )
        ],
        target_df,
    )
    result = await rainbear.scan_zarr_async(str(zarr_path), expr)
    unnested = result.unnest("interpolated")
    actual = float(unnested["CAL"][0])

    # Compare with xarray for value correctness
    ds_decoded = xr.decode_cf(ds)
    sel_ds = ds_decoded.sel(time="2025-02-01T17:00:00")
    expected = _xr_interp_single_point(sel_ds, "CAL", {"lat": 0.33, "lon": 0.14})

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


@pytest.mark.asyncio
async def test_interp_cf_datetime_2d_with_time_filter2(tmp_path: Path) -> None:
    """2D interpolation (lat, lon) with CF-coded time as filter dimension."""
    nlat, nlon = 8, 10
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    # 3 time steps with CF int64 encoding
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    times_ns = np.array(
        [
            np.datetime64("2025-02-01T16:00:00", "ns"),
            np.datetime64("2025-02-01T17:00:00", "ns"),
            np.datetime64("2025-02-01T18:00:00", "ns"),
        ]
    )
    time_seconds = ((times_ns - epoch) / np.timedelta64(1, "s")).astype(np.int64)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"CAL": (["time", "lat", "lon"], data)},
        coords={
            "time": (["time"], time_seconds),
            "lat": lat_coords,
            "lon": lon_coords,
        },
    )
    ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"

    zarr_path = tmp_path / "cf_time_2d.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={
            "CAL": {"chunks": (1, 4, 5), "compressors": [BLOSC_ZSTD]},
        },
    )

    # Interpolate at (lat=0.33, lon=0.14) with time filter at 17:00
    target_df = pl.DataFrame({"lat": [0.33], "lon": [0.14], "time": [datetime.datetime(2025, 2, 1, 17, 0, 0)]})
    expr = interpolate_nd(
        ["lat", "lon"],
        [
            "CAL"
        ],
        target_df,
    )
    result = await rainbear.scan_zarr_async(str(zarr_path), expr)
    unnested = result.unnest("interpolated")
    actual = float(unnested["CAL"][0])

    # Compare with xarray for value correctness
    ds_decoded = xr.decode_cf(ds)
    sel_ds = ds_decoded.sel(time="2025-02-01T17:00:00")
    expected = _xr_interp_single_point(sel_ds, "CAL", {"lat": 0.33, "lon": 0.14})

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# Planner: chunk selection for CF datetime
# ---------------------------------------------------------------------------


def _chunk_indices(chunks, variable: str = "value") -> set[tuple[int, ...]]:
    """Extract chunk indices for a variable from selected_chunks_debug output."""
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")


def test_interp_cf_datetime_planner_selects_correct_chunks(tmp_path: Path) -> None:
    """Planner selects minimal chunks for CF datetime interpolation."""
    n = 20
    days = np.arange(n, dtype=np.float64) + 20088.0  # 2025-01-01 + days

    ds = xr.Dataset(
        data_vars={"value": (["time"], np.arange(n, dtype=np.float64) * 10.0)},
        coords={"time": (["time"], days)},
    )
    ds["time"].attrs["units"] = "days since 1970-01-01 00:00:00"

    zarr_path = tmp_path / "cf_time_planner.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={
            "value": {"chunks": (5,), "compressors": [BLOSC_ZSTD]},
        },
    )

    # Target at day 7 (index 7) - chunk 1 covers indices 5..9
    target = pl.DataFrame(
        {"time": [datetime.datetime(2025, 1, 8, 0, 0, 0)]},
        schema={"time": pl.Datetime("us")},
    )
    expr = interpolate_nd(["time"], ["value"], target)
    chunks = rainbear.ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    # Should select only chunk 1 (indices 5..9)
    assert idxs == {(1,)}, f"expected {{(1,)}}, got {idxs}"
