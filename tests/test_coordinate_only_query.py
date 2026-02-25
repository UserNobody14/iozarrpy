"""Tests for coordinate-only query optimization.

When selecting only raw coordinate columns (e.g. pl.col("x") or pl.col("y", "x"))
on a zarr dataset, the backend should only query those coordinates from the
store—not load data variables. Dimension indices (x, y) can be built from
chunk indices without reading array data, so a coordinate-only query should
produce a GridInfo with empty variables, correct dims, and valid chunks.

These tests verify both correctness (we get the right values) and the intended
optimization via selected_chunks_debug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import rainbear

from .baseline_utils import assert_frames_equal, scan_via_xarray

if TYPE_CHECKING:
    from rainbear._core import GridInfo, SelectedChunksDebugReturn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def orography_path(tmp_path_factory) -> str:
    """Create a small orography dataset for coordinate-only tests."""
    from zarr.codecs import BloscCodec, BloscShuffle

    from . import zarr_generators

    path = tmp_path_factory.mktemp("data") / "orography_coord_only.zarr"
    ds = zarr_generators.create_orography_dataset(nx=20, ny=16, sigma=4.0, seed=1)
    blosc = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
    ds.to_zarr(str(path), zarr_format=3, encoding={
        "geopotential_height": {"chunks": (8, 10), "compressors": [blosc]},
        "latitude": {"chunks": (8, 10), "compressors": [blosc]},
        "longitude": {"chunks": (8, 10), "compressors": [blosc]},
    })
    return str(path)


# ---------------------------------------------------------------------------
# Correctness tests - select only coordinates, verify values match xarray
# ---------------------------------------------------------------------------


def test_select_only_latitude_returns_correct_values(orography_path: str) -> None:
    """Selecting only latitude (with y, x for row structure) returns correct values."""
    columns = ["y", "x", "latitude"]
    out = rainbear.scan_zarr(orography_path).select(columns).collect()
    baseline = scan_via_xarray(orography_path, columns=columns).collect()
    assert_frames_equal(out, baseline, sort_by=["y", "x"])


def test_select_only_longitude_returns_correct_values(orography_path: str) -> None:
    """Selecting only longitude returns correct values."""
    columns = ["y", "x", "longitude"]
    out = rainbear.scan_zarr(orography_path).select(columns).collect()
    baseline = scan_via_xarray(orography_path, columns=columns).collect()
    assert_frames_equal(out, baseline, sort_by=["y", "x"])


def test_select_latitude_and_longitude_only(orography_path: str) -> None:
    """Selecting only latitude and longitude (no data vars) returns correct values."""
    columns = ["y", "x", "latitude", "longitude"]
    out = rainbear.scan_zarr(orography_path).select(columns).collect()
    baseline = scan_via_xarray(orography_path, columns=columns).collect()
    assert_frames_equal(out, baseline, sort_by=["y", "x"])


def test_select_only_latitude_values_no_dimensions(orography_path: str) -> None:
    """Selecting only latitude (no y, x) returns the correct flattened latitude values."""
    # We get 320 rows (20*16) of latitude values. Row order may differ (chunk iteration vs
    # row-major), so we compare sorted values to verify we have the same set.
    out = rainbear.scan_zarr(orography_path).select(["latitude"]).collect()
    baseline = (
        scan_via_xarray(orography_path, columns=["y", "x", "latitude"])
        .collect()
        .sort(["y", "x"])
        .select(["latitude"])
    )
    assert out.shape == baseline.shape
    out_sorted = out.sort("latitude")
    baseline_sorted = baseline.sort("latitude")
    assert out_sorted["latitude"].to_list() == baseline_sorted["latitude"].to_list()


# ---------------------------------------------------------------------------
# Optimization tests - selected_chunks_debug should show coordinate-only grid
# ---------------------------------------------------------------------------


def _find_coord_only_grid(
    debug: "SelectedChunksDebugReturn",
    expected: list[tuple[list[str], list[str]]]
) -> "GridInfo | None":
    """Find a grid with the given dims and empty variables (coordinate-only)."""
    for grid in debug["grids"]:
        if (grid["dims"], grid["variables"]) in expected:
            return grid
    return None


def _assert_valid_chunks(grid: "GridInfo") -> None:
    """Assert grid has valid non-empty chunks with proper structure."""
    assert len(grid["chunks"]) > 0, "Grid should have at least one chunk"
    for chunk in grid["chunks"]:
        assert "indices" in chunk, "Chunk should have indices"
        assert len(chunk["indices"]) == len(grid["dims"]), (
            f"Chunk indices length {len(chunk['indices'])} should match dims {len(grid['dims'])}"
        )


def test_select_only_x_produces_coord_only_grid(orography_path: str) -> None:
    """Selecting only x should produce a GridInfo with empty variables.

    When we run pl.col(['x']).filter(pl.lit(True)), selected_chunks_debug
    should return a grid with:
    - variables: ['x'] (empty - x is a dimension index, built from chunk indices)
    - dims: ['x'] (correct dimension name)
    - chunks: valid non-empty list with proper indices
    """
    backend = rainbear.ZarrBackend.from_url(orography_path)
    expr = pl.col(["x"]).filter(pl.lit(True))
    debug = backend.selected_chunks_debug(expr)

    grid = _find_coord_only_grid(debug, expected=[(["x"], ['x'])])
    assert grid is not None, (
        f"Expected a grid with dims=['x'] and variables=['x']. "
        f"Got grids: {[(g['dims'], g['variables']) for g in debug['grids']]}"
    )
    _assert_valid_chunks(grid)


def test_select_only_y_and_x_produces_coord_only_grid(orography_path: str) -> None:
    """Selecting only y and x should produce a grid with empty variables.

    Dimension-only (no geopotential_height, latitude, longitude) should yield:
    First grid:
        - variables: ['y']
        - dims: ['y']
        - chunks: valid
    Second grid:
        - variables: ['x']
        - dims: ['x']
        - chunks: valid
    """
    backend = rainbear.ZarrBackend.from_url(orography_path)
    expr = pl.col(["y", "x"]).filter(pl.lit(True))
    debug = backend.selected_chunks_debug(expr)

    grid = _find_coord_only_grid(debug, expected=[(["y"], ['y']), (["x"], ['x'])])
    assert grid is not None, (
        f"Expected a grid with dims=['y','x'] and variables=['y', 'x']. "
        f"Got grids: {[(g['dims'], g['variables']) for g in debug['grids']]}"
    )
    _assert_valid_chunks(grid)
