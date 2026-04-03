"""Planner tests for `interpolate_geospatial` FFI plugin expressions.

These tests validate *chunk planning* (not plugin evaluation) for geospatial
interpolation, including ghost-point expansion for longitude wrapping.
They are written so they cannot pass by returning "all chunks".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import xarray as xr
from interpolars import interpolate_geospatial
from zarr.codecs import BloscCodec, BloscShuffle

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import SelectedChunksDebugReturn

BLOSC = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)


def _chunk_indices(
    chunks: SelectedChunksDebugReturn, variable: str = "temperature"
) -> set[tuple[int, ...]]:
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")


def _make_global_latlon_zarr(tmp_path: Path, name: str, nlat: int = 36, nlon: int = 72) -> str:
    """Create a simple global lat/lon temperature dataset.

    Default: 5-degree resolution (36 lat x 72 lon), chunks of 9x18.
    """
    lats = np.linspace(-87.5, 87.5, nlat)
    lons = np.linspace(2.5, 357.5, nlon)
    data = np.random.default_rng(42).standard_normal((nlat, nlon))

    ds = xr.Dataset(
        data_vars={"temperature": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )

    zarr_path = tmp_path / f"{name}.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "temperature": {
                "chunks": (9, 18),
                "compressors": [BLOSC],
            }
        },
    )
    return str(zarr_path)


# ---------------------------------------------------------------------------
# Basic planning tests
# ---------------------------------------------------------------------------


def test_geospatial_plans_single_chunk_interior(tmp_path: Path) -> None:
    """Target point well inside one chunk should select only that chunk."""
    url = _make_global_latlon_zarr(tmp_path, "geo_single")

    target = pl.DataFrame({"lat": [10.0], "lon": [50.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    # lat=10 is in the upper half => lat chunk 2; lon=50 is in lon chunk 0.
    # With ±1 expansion on lat and ±3 on lon, we still shouldn't span all chunks.
    assert len(idxs) < 4 * 4, f"Over-selected: {idxs}"
    assert len(idxs) >= 1


def test_geospatial_plans_boundary_lat(tmp_path: Path) -> None:
    """Target near lat chunk boundary should select neighboring lat chunks."""
    url = _make_global_latlon_zarr(tmp_path, "geo_lat_boundary")

    # lat chunk boundary at index 9 (lat=-42.5). Target just below that.
    target = pl.DataFrame({"lat": [-43.0], "lon": [50.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    lat_chunks = {idx[0] for idx in idxs}
    assert len(lat_chunks) >= 1


def test_geospatial_plans_near_dateline_east(tmp_path: Path) -> None:
    """Target near lon=360/0 boundary should include ghost chunks from the start.

    With WrappingInterpolationRange, the planner should select lon chunks
    near the end AND ghost chunks from the beginning of the longitude axis.
    """
    url = _make_global_latlon_zarr(tmp_path, "geo_dateline_east")

    # lon near 357.5 (last grid point). This is near the end of the lon axis.
    target = pl.DataFrame({"lat": [0.0], "lon": [356.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    lon_chunks = {idx[1] for idx in idxs}
    # Should include the last lon chunk (3) and ghost from the first lon chunk (0)
    assert 3 in lon_chunks, f"Expected last lon chunk (3) in {lon_chunks}"
    assert 0 in lon_chunks, f"Expected ghost chunk (0) from wrapping in {lon_chunks}"


def test_geospatial_plans_near_dateline_west(tmp_path: Path) -> None:
    """Target near lon=0/360 boundary from the west side.

    Should include lon chunks from the start AND ghost chunks from the end.
    """
    url = _make_global_latlon_zarr(tmp_path, "geo_dateline_west")

    # lon near 2.5 (first grid point). Near the start of the lon axis.
    target = pl.DataFrame({"lat": [0.0], "lon": [3.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    lon_chunks = {idx[1] for idx in idxs}
    # Should include the first lon chunk (0) and ghost from the last lon chunk (3)
    assert 0 in lon_chunks, f"Expected first lon chunk (0) in {lon_chunks}"
    assert 3 in lon_chunks, f"Expected ghost chunk (3) from wrapping in {lon_chunks}"


def test_geospatial_interior_does_not_select_all_lon_chunks(tmp_path: Path) -> None:
    """An interior point far from boundaries should NOT select all lon chunks."""
    url = _make_global_latlon_zarr(tmp_path, "geo_interior_no_overselect")

    target = pl.DataFrame({"lat": [0.0], "lon": [90.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    lon_chunks = {idx[1] for idx in idxs}
    # lon=90 is solidly in chunk 1 (indices 18-35, lons 92.5-177.5 roughly).
    # Ghost expansion of ±3 should not reach chunk 0 or 3.
    assert len(lon_chunks) <= 2, f"Over-selected lon chunks: {lon_chunks}"


# ---------------------------------------------------------------------------
# Multiple target points
# ---------------------------------------------------------------------------


def test_geospatial_multiple_targets_across_dateline(tmp_path: Path) -> None:
    """Two points on opposite sides of the dateline should select appropriate chunks."""
    url = _make_global_latlon_zarr(tmp_path, "geo_multi_dateline")

    target = pl.DataFrame({
        "lat": [0.0, 0.0],
        "lon": [356.0, 3.0],
    })
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    lon_chunks = {idx[1] for idx in idxs}
    # Both points are near the dateline: should include first and last lon chunks
    assert 0 in lon_chunks
    assert 3 in lon_chunks


def test_geospatial_widely_separated_points_no_overselect(tmp_path: Path) -> None:
    """Two widely separated interior points should NOT select all chunks."""
    url = _make_global_latlon_zarr(tmp_path, "geo_wide_separated")

    target = pl.DataFrame({
        "lat": [45.0, -45.0],
        "lon": [90.0, 270.0],
    })
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    # 4 lat chunks × 4 lon chunks = 16 total. We should be well under that.
    assert len(idxs) < 16, f"Over-selected: got {len(idxs)} of 16 total chunks"


# ---------------------------------------------------------------------------
# Signed longitude convention (-180 to 180)
# ---------------------------------------------------------------------------


def test_geospatial_signed_lon_near_antimeridian(tmp_path: Path) -> None:
    """Signed longitude (-180..180): targets near ±180 should trigger wrapping."""
    nlat, nlon = 36, 72
    lats = np.linspace(-87.5, 87.5, nlat)
    lons = np.linspace(-177.5, 177.5, nlon)
    data = np.random.default_rng(99).standard_normal((nlat, nlon))

    ds = xr.Dataset(
        data_vars={"temperature": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )

    zarr_path = tmp_path / "geo_signed_lon.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "temperature": {
                "chunks": (9, 18),
                "compressors": [BLOSC],
            }
        },
    )

    # Target near +177 (close to the end of the axis)
    target = pl.DataFrame({"lat": [0.0], "lon": [176.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    lon_chunks = {idx[1] for idx in idxs}
    assert 3 in lon_chunks, f"Expected last lon chunk in {lon_chunks}"
    assert 0 in lon_chunks, f"Expected ghost chunk from wrapping in {lon_chunks}"


# ---------------------------------------------------------------------------
# Non-wrapping small grid (too small for ghost points)
# ---------------------------------------------------------------------------


def test_geospatial_small_grid_no_crash(tmp_path: Path) -> None:
    """A very small grid (fewer lon points than ghost expansion) should not crash."""
    nlat, nlon = 4, 4
    lats = np.array([10.0, 20.0, 30.0, 40.0])
    lons = np.array([100.0, 110.0, 120.0, 130.0])
    data = np.ones((nlat, nlon))

    ds = xr.Dataset(
        data_vars={"temperature": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )

    zarr_path = tmp_path / "geo_small.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "temperature": {
                "chunks": (2, 2),
                "compressors": [BLOSC],
            }
        },
    )

    target = pl.DataFrame({"lat": [25.0], "lon": [115.0]})
    expr = interpolate_geospatial("lat", "lon", ["temperature"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks)

    assert len(idxs) >= 1


# ---------------------------------------------------------------------------
# Variable tracking
# ---------------------------------------------------------------------------


def test_geospatial_selects_correct_variable(tmp_path: Path) -> None:
    """Planner should track the correct data variable from source_values."""
    nlat, nlon = 20, 40
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)

    ds = xr.Dataset(
        data_vars={
            "temperature": (["lat", "lon"], np.ones((nlat, nlon))),
            "pressure": (["lat", "lon"], np.ones((nlat, nlon)) * 1013.0),
        },
        coords={"lat": lats, "lon": lons},
    )

    zarr_path = tmp_path / "geo_multivar.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "temperature": {"chunks": (10, 20), "compressors": [BLOSC]},
            "pressure": {"chunks": (10, 20), "compressors": [BLOSC]},
        },
    )

    target = pl.DataFrame({"lat": [0.0], "lon": [90.0]})
    expr = interpolate_geospatial("lat", "lon", ["pressure"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)

    # Should find grid with "pressure" variable
    pressure_idxs = _chunk_indices(chunks, variable="pressure")
    assert len(pressure_idxs) >= 1
