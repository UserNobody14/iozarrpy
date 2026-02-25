"""Interpolation value correctness tests: rainbear interpolate_nd vs xarray .interp().

These tests compare the actual interpolated values produced by rainbear's
``interpolate_nd`` plugin against xarray's ``.interp()`` (which uses scipy
linear interpolation under the hood).  They are parameterised over several
dataset shapes and target points.

The bug being captured: ``interpolate_nd`` produces incorrect results when the
coordinate arrays are **non-integer** (e.g. ``np.linspace(-0.5, 1.5, 10)``).
Integer coordinate grids (0, 1, 2, …) interpolate correctly, but arbitrary
floating-point coordinates do not - the weights appear to be computed from
array indices rather than from the actual coordinate values.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xr_interp_single_point(
    ds: xr.Dataset,
    var: str,
    coords: dict[str, float],
) -> float:
    """Return the scalar result of xarray linear interpolation at one point."""
    kw = {k: [v] for k, v in coords.items()}
    result = ds.interp(**kw).compute()
    return float(result[var].values.flat[0])


async def _rb_interp_single_point(
    zarr_path: str,
    coord_names: list[str],
    value_exprs: list,
    coords: dict[str, float],
) -> pl.DataFrame:
    """Run rainbear interpolate_nd and return the unnested result."""
    target = pl.DataFrame(coords)
    expr = interpolate_nd(coord_names, value_exprs, target)
    result = await rainbear.scan_zarr_async(zarr_path, expr)
    return result.unnest("interpolated")


# ---------------------------------------------------------------------------
# 1-D  non-uniform coordinates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interp_1d_nonuniform_coords(tmp_path: Path) -> None:
    """1D interpolation with non-integer coordinate spacing."""
    n = 20
    x_coords = np.linspace(0.5, 10.5, n)
    rng = np.random.default_rng(99)
    values = rng.standard_normal(n).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"value": (["x"], values)},
        coords={"x": x_coords},
    )

    zarr_path = tmp_path / "interp_1d_nonuniform.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"value": {"chunks": (10,), "compressors": [BLOSC_ZSTD]}},
    )

    target_x = 1.2
    expected = _xr_interp_single_point(ds, "value", {"x": target_x})

    rb = await _rb_interp_single_point(
        str(zarr_path), ["x"], ["value"], {"x": target_x}
    )
    actual = float(rb["value"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# 2-D  non-uniform coordinates  (the core bug)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interp_2d_nonuniform_coords_single_point(tmp_path: Path) -> None:
    """2D bilinear interpolation with non-integer coordinate spacing."""
    nlat, nlon = 10, 12
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"value": (["lat", "lon"], data)},
        coords={"lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_2d_nonuniform.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"value": {"chunks": (5, 6), "compressors": [BLOSC_ZSTD]}},
    )

    target = {"lat": 0.33, "lon": 0.14}
    expected = _xr_interp_single_point(ds, "value", target)

    rb = await _rb_interp_single_point(
        str(zarr_path), ["lat", "lon"], ["value"], target
    )
    actual = float(rb["value"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


@pytest.mark.asyncio
async def test_interp_2d_nonuniform_coords_single_chunk(tmp_path: Path) -> None:
    """Same as above but with all data in one chunk (rules out chunking issues)."""
    nlat, nlon = 10, 12
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"value": (["lat", "lon"], data)},
        coords={"lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_2d_singlechunk.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"value": {"chunks": (nlat, nlon), "compressors": [BLOSC_ZSTD]}},
    )

    target = {"lat": 0.33, "lon": 0.14}
    expected = _xr_interp_single_point(ds, "value", target)

    rb = await _rb_interp_single_point(
        str(zarr_path), ["lat", "lon"], ["value"], target
    )
    actual = float(rb["value"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lat_pt, lon_pt",
    [
        (0.33, 0.14),
        (0.5, 1.0),
        (1.0, 2.5),
        (-0.25, 0.75),
        (0.0, 0.0),
    ],
    ids=["interior_a", "interior_b", "interior_c", "near_lower_bound", "at_grid_node"],
)
async def test_interp_2d_nonuniform_coords_multiple_points(
    tmp_path: Path, lat_pt: float, lon_pt: float
) -> None:
    """Parameterised over several target points."""
    nlat, nlon = 10, 12
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"value": (["lat", "lon"], data)},
        coords={"lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_2d_param.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"value": {"chunks": (5, 6), "compressors": [BLOSC_ZSTD]}},
    )

    target = {"lat": lat_pt, "lon": lon_pt}
    expected = _xr_interp_single_point(ds, "value", target)

    rb = await _rb_interp_single_point(
        str(zarr_path), ["lat", "lon"], ["value"], target
    )
    actual = float(rb["value"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"point=({lat_pt}, {lon_pt}): rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# 2-D  integer coordinates  (sanity check - should pass)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interp_2d_integer_coords_passes(tmp_path: Path) -> None:
    """Integer coordinates should produce correct results (baseline sanity)."""
    nlat, nlon = 10, 12
    lat_coords = np.arange(nlat, dtype=np.float64)
    lon_coords = np.arange(nlon, dtype=np.float64)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"value": (["lat", "lon"], data)},
        coords={"lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_2d_intcoords.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"value": {"chunks": (5, 6), "compressors": [BLOSC_ZSTD]}},
    )

    target = {"lat": 2.5, "lon": 3.5}
    expected = _xr_interp_single_point(ds, "value", target)

    rb = await _rb_interp_single_point(
        str(zarr_path), ["lat", "lon"], ["value"], target
    )
    actual = float(rb["value"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# 3-D  with time filter  (mirrors the notebook pattern)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interp_3d_with_time_filter_nonuniform_coords(tmp_path: Path) -> None:
    """3D dataset with time filter - mirrors the notebook's CAL interpolation."""
    nt, nlat, nlon = 3, 10, 12
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)
    time_coords = np.array(
        ["2025-02-01T16:00", "2025-02-01T17:00", "2025-02-01T18:00"],
        dtype="datetime64[ns]",
    )

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nt, nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"CAL": (["time", "lat", "lon"], data)},
        coords={"time": time_coords, "lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_3d_time_filter.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={
            "CAL": {"chunks": (1, 5, 6), "compressors": [BLOSC_ZSTD]},
        },
    )

    sel_ds = ds.sel(time="2025-02-01T17:00")
    target = {"lat": 0.33, "lon": 0.14}
    expected = _xr_interp_single_point(sel_ds, "CAL", target)

    target_df = pl.DataFrame(target)
    expr = interpolate_nd(
        ["lat", "lon"],
        [
            pl.col("CAL").filter(
                pl.col("time") == datetime.datetime(2025, 2, 1, 17)
            )
        ],
        target_df,
    )
    result = await rainbear.scan_zarr_async(str(zarr_path), expr)
    unnested = result.unnest("interpolated")
    actual = float(unnested["CAL"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, xarray={expected}"
    )


# ---------------------------------------------------------------------------
# Linear ramp  (analytically verifiable)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interp_2d_quadratic_nonuniform_coords(tmp_path: Path) -> None:
    """Quadratic f(lat,lon) = lat * lon with non-integer coords.

    A multiplicative function exercises bilinear weight correctness:
    unlike an additive function (lat + lon), bilinear interpolation of
    lat * lon is sensitive to the actual interpolation weights.

    Uses small-range non-integer coordinates matching the patterns that
    trigger the interpolation bug (e.g. lat/lon in [-0.5, 3.5]).
    """
    nlat, nlon = 8, 10
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    lats, lons = np.meshgrid(lat_coords, lon_coords, indexing="ij")
    data = lats * lons

    ds = xr.Dataset(
        data_vars={"product": (["lat", "lon"], data)},
        coords={"lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_2d_quad.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"product": {"chunks": (4, 5), "compressors": [BLOSC_ZSTD]}},
    )

    lat_pt, lon_pt = 0.33, 0.14
    expected = _xr_interp_single_point(
        ds, "product", {"lat": lat_pt, "lon": lon_pt}
    )

    rb = await _rb_interp_single_point(
        str(zarr_path),
        ["lat", "lon"],
        ["product"],
        {"lat": lat_pt, "lon": lon_pt},
    )
    actual = float(rb["product"][0])

    assert actual == pytest.approx(expected, abs=1e-6), (
        f"rainbear={actual}, xarray={expected}"
    )


@pytest.mark.asyncio
async def test_interp_1d_linear_ramp_nonuniform_coords(tmp_path: Path) -> None:
    """1D linear ramp f(x) = 2*x + 5 with non-integer coords."""
    n = 15
    x_coords = np.linspace(1.0, 8.0, n)
    values = 2.0 * x_coords + 5.0

    ds = xr.Dataset(
        data_vars={"ramp": (["x"], values)},
        coords={"x": x_coords},
    )

    zarr_path = tmp_path / "interp_1d_ramp.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"ramp": {"chunks": (8,), "compressors": [BLOSC_ZSTD]}},
    )

    x_pt = 4.37
    expected = 2.0 * x_pt + 5.0

    rb = await _rb_interp_single_point(
        str(zarr_path), ["x"], ["ramp"], {"x": x_pt}
    )
    actual = float(rb["ramp"][0])

    assert actual == pytest.approx(expected, abs=1e-10), (
        f"rainbear={actual}, expected={expected}"
    )
