"""Planner tests for `interpolate_nd` FFI plugin expressions.

These tests only validate *chunk planning* (not plugin evaluation), and are written so they
cannot pass by returning “all chunks”.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from interpolars import interpolate_nd
from zarr.codecs import BloscCodec, BloscShuffle

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import SelectedChunksDebugReturn


def _chunk_indices(chunks: SelectedChunksDebugReturn, variable: str = "geopotential_height") -> set[tuple[int, ...]]:
    # Find a grid that includes the variable
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")


def test_interpolate_nd_plans_single_chunk(baseline_datasets: dict[str, str], tmp_path: Path) -> None:
    zarr_url = baseline_datasets["orography_chunked_10x10"]

    target = pl.DataFrame(
        {
            "y": [0, 5, 8],
            "x": [0, 4, 8],
            "labels": ["a", "b", "c"],
        }
    )
    expr = interpolate_nd(["y", "x"], ["geopotential_height"], target)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks)

    # Grid is 2x2 for (y,x) with chunks (10,10) and shape (16,20) => 4 chunks total.
    assert idxs == {(0, 0)}


def test_interpolate_nd_plans_two_chunks_across_x_boundary(
    baseline_datasets: dict[str, str], tmp_path: Path
) -> None:
    zarr_url = baseline_datasets["orography_chunked_10x10"]

    target = pl.DataFrame(
        {
            "y": [0, 8],
            "x": [9, 11],
            "labels": ["a", "b"],
        }
    )
    expr = interpolate_nd(["y", "x"], ["geopotential_height"], target)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks)

    # y stays in chunk 0, x straddles boundary at 10 => chunks (0,0) and (0,1).
    assert idxs == {(0, 0), (0, 1)}


def test_interpolate_nd_groups_by_extra_columns_does_not_overselect(tmp_path: Path) -> None:
    """When interpolars groups by extra coord columns, we should union chunks per-point.

    This test intentionally uses widely separated x values so a naive [min,max] range
    would select *all* x chunks between them.
    """
    import xarray as xr

    # Build a dataset with many x chunks (100 x-values, chunks of 10 => 10 chunks).
    nx, ny = 100, 16
    ds = xr.Dataset(
        data_vars={
            "geopotential_height": (["y", "x"], [[float(i + j) for i in range(nx)] for j in range(ny)]),
        },
        coords={
            "y": list(range(ny)),
            "x": list(range(nx)),
        },
    )

    zarr_path = tmp_path / "interp_groups.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "geopotential_height": {
                "chunks": (10, 10),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # Extra column `label` is not part of coords list => interpolars will group by it.
    # Two groups with x-values far apart: 5 (chunk 0) and 95 (chunk 9).
    target = pl.DataFrame(
        {
            "y": [0, 0],
            "x": [5, 95],
            "label": ["a", "b"],
        }
    )

    expr = interpolate_nd(["y", "x"], ["geopotential_height"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks)

    # y is in chunk 0; x touches chunks 0 and 9 (plus interpolation neighbors stay within those chunks).
    assert idxs == {(0, 0), (0, 9)}


def test_interpolate_nd_group_dim_in_source_coords_is_unconstrained_but_keeps_xy_constraints(
    tmp_path: Path,
) -> None:
    """If `time` is in source coords but absent from target, it's a group key.

    Planner should still prune on y/x, while leaving time unconstrained (selecting all time chunks).
    """
    import numpy as np
    import xarray as xr

    nt, ny, nx = 2, 16, 20
    data = np.arange(nt * ny * nx, dtype=np.float64).reshape(nt, ny, nx)
    ds = xr.Dataset(
        data_vars={"geopotential_height": (["time", "y", "x"], data)},
        coords={"time": [0, 1], "y": list(range(ny)), "x": list(range(nx))},
    )

    zarr_path = tmp_path / "interp_group_time.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "geopotential_height": {
                "chunks": (1, 10, 10),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # Target has no `time` => interpolars groups by `time` (two groups).
    target = pl.DataFrame({"y": [0], "x": [5], "label": ["only"]})
    expr = interpolate_nd(["y", "x", "time"], ["geopotential_height"], target)

    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks)

    # y=0 is in y-chunk 0; x=5 in x-chunk 0. time unconstrained => both time chunks.
    assert idxs == {(0, 0, 0), (1, 0, 0)}


def test_interpolate_nd_filtered_source_values_constrains_time(tmp_path: Path) -> None:
    """When source_values uses col("X").filter(time == t), planner constrains time dimension."""
    import numpy as np
    import xarray as xr

    nt, ny, nx = 3, 16, 20
    data = np.arange(nt * ny * nx, dtype=np.float64).reshape(nt, ny, nx)
    ds = xr.Dataset(
        data_vars={"value": (["time", "y", "x"], data)},
        coords={"time": [0, 1, 2], "y": list(range(ny)), "x": list(range(nx))},
    )

    zarr_path = tmp_path / "interp_filtered.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (1, 10, 10),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    target = pl.DataFrame({"y": [0], "x": [5]})
    # Filter constrains time to 1 => only time chunk 1
    expr = interpolate_nd(
        ["y", "x"],
        [pl.col("value").filter(pl.col("time") == 1)],
        target,
    )
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    assert idxs == {(1, 0, 0)}


def test_interpolate_nd_out_of_bounds_targets_clamp_to_boundary_chunks(tmp_path: Path) -> None:
    """README says out-of-bounds targets clamp; planner must include boundary chunks."""
    import xarray as xr

    nx, ny = 100, 16
    ds = xr.Dataset(
        data_vars={
            "geopotential_height": (["y", "x"], [[float(i + j) for i in range(nx)] for j in range(ny)]),
        },
        coords={"y": list(range(ny)), "x": list(range(nx))},
    )

    zarr_path = tmp_path / "interp_oob_clamp.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "geopotential_height": {
                "chunks": (10, 10),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # x=-5 clamps to 0 (chunk 0); x=150 clamps to 99 (chunk 9).
    target = pl.DataFrame({"y": [0, 0], "x": [-5, 150], "label": ["lo", "hi"]})
    expr = interpolate_nd(["y", "x"], ["geopotential_height"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks)

    assert idxs == {(0, 0), (0, 9)}


def test_interpolate_nd_date_coords_plan_and_clamp(tmp_path: Path) -> None:
    """Validate planning when the interpolation axis is Date-like."""
    import numpy as np
    import xarray as xr

    n = 100
    d = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n)])
    ds = xr.Dataset(
        data_vars={"value": (["d"], np.arange(n, dtype=np.float64))},
        coords={"d": d},
    )

    zarr_path = tmp_path / "interp_date_1d.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (10,),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    target = pl.DataFrame(
        {
            "d": pl.Series("d", [date(1900, 1, 1), date(2100, 1, 1)], dtype=pl.Date),
            "label": ["lo", "hi"],
        }
    )

    expr = interpolate_nd(["d"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks, variable="value")

    # Out-of-bounds clamps to start/end, so we only need first + last chunk.
    assert idxs == {(0,), (9,)}


def test_interpolate_nd_duration_coords_plan_and_clamp(tmp_path: Path) -> None:
    """Validate planning when the interpolation axis is Duration-like."""
    import numpy as np
    import xarray as xr

    n = 100
    # 0s..99s duration axis.
    dt = np.array([np.timedelta64(i, "s") for i in range(n)], dtype="timedelta64[ns]")
    ds = xr.Dataset(
        data_vars={"value": (["dt"], np.arange(n, dtype=np.float64))},
        coords={"dt": dt},
    )

    zarr_path = tmp_path / "interp_duration_1d.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (10,),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    target = pl.DataFrame(
        {
            "dt": pl.Series(
                "dt",
                [-5_000, 150_000],
                dtype=pl.Duration("ms"),
            ),
            "label": ["lo", "hi"],
        }
    )

    expr = interpolate_nd(["dt"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug( expr)
    idxs = _chunk_indices(chunks, variable="value")

    assert idxs == {(0,), (9,)}


# ---------------------------------------------------------------------------
# Sparse / non-uniform coordinate tests
# ---------------------------------------------------------------------------
# The interpolation planner uses resolution to find bracketing indices (left_idx, right_idx)
# for each target point. It expands by ±1 in index space for chunk boundaries. These tests
# verify correct behavior when coordinate *values* jump by large amounts (e.g. [0, 100, 200])
# rather than consecutive integers. The bracket indices come from binary search on the coord
# array, so they should be correct regardless of value spacing; these tests document and
# verify that behavior.


def test_interpolate_nd_sparse_1d_coords_bracket_within_chunk(tmp_path: Path) -> None:
    """1D sparse coords: interpolation point between indices in same chunk.

    Coords [0, 100, 200, ..., 1900] (20 points). Chunk size 5 => 4 chunks.
    Interpolate at 550: bracket is indices 5 and 6 (coords 500, 600). Both in chunk 1.
    Planner may conservatively include chunk 0 due to ±1 expansion; we assert at least chunk 1.
    """
    import numpy as np
    import xarray as xr

    n = 20
    coords = np.array([i * 100 for i in range(n)], dtype=np.float64)
    ds = xr.Dataset(
        data_vars={"value": (["x"], np.arange(n, dtype=np.float64))},
        coords={"x": coords},
    )

    zarr_path = tmp_path / "interp_sparse_1d.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (5,),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # 550 is between coord 500 (idx 5) and 600 (idx 6). Both in chunk 1 (indices 5-9).
    target = pl.DataFrame({"x": [550.0], "label": ["mid"]})
    expr = interpolate_nd(["x"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    # Must include chunk 1 (bracket indices 5,6). Expansion may add chunk 0.
    assert (1,) in idxs


def test_interpolate_nd_sparse_1d_coords_bracket_spans_chunk_boundary(tmp_path: Path) -> None:
    """1D sparse coords: interpolation point where bracket spans chunk boundary.

    Coords [0, 10, 20, ..., 390] (40 points). Chunk size 10 => 4 chunks.
    Interpolate at 95: bracket is indices 9 and 10 (coords 90, 100). Index 9 in chunk 0,
    index 10 in chunk 1. Must select both chunks.
    """
    import numpy as np
    import xarray as xr

    n = 40
    coords = np.array([i * 10.0 for i in range(n)], dtype=np.float64)
    ds = xr.Dataset(
        data_vars={"value": (["x"], np.arange(n, dtype=np.float64))},
        coords={"x": coords},
    )

    zarr_path = tmp_path / "interp_sparse_1d_boundary.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (10,),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # 95 is between coord 90 (idx 9) and 100 (idx 10). Span chunk 0 and 1.
    target = pl.DataFrame({"x": [95.0], "label": ["boundary"]})
    expr = interpolate_nd(["x"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    assert idxs == {(0,), (1,)}


def test_interpolate_nd_sparse_1d_large_coord_gaps(tmp_path: Path) -> None:
    """1D: coordinate values jump by 1000; bracket spans exactly two chunks.

    Coords [0, 1000, 2000, 3000, 4000] (5 points). Chunk size 1 => 5 chunks.
    Interpolate at 500: needs indices 0 and 1. Interpolate at 2500: needs indices 2 and 3.
    Per-point selection should NOT overselect; we get only the 2 chunks per point.
    """
    import numpy as np
    import xarray as xr

    n = 5
    coords = np.array([i * 1000.0 for i in range(n)], dtype=np.float64)
    ds = xr.Dataset(
        data_vars={"value": (["x"], np.arange(n, dtype=np.float64))},
        coords={"x": coords},
    )

    zarr_path = tmp_path / "interp_sparse_large_gaps.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (1,),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # Two points: 500 (bracket 0,1) and 2500 (bracket 2,3). Should get chunks 0,1,2,3 (not 4).
    target = pl.DataFrame({"x": [500.0, 2500.0], "label": ["early", "late"]})
    expr = interpolate_nd(["x"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    # Per-point: 500 needs chunks 0,1; 2500 needs chunks 2,3. Union = {0,1,2,3}. Chunk 4 unused.
    assert idxs == {(0,), (1,), (2,), (3,)}


def test_interpolate_nd_sparse_2d_coords_bracket_within_chunk(tmp_path: Path) -> None:
    """2D sparse coords: interpolation point where both dim brackets stay in one chunk.

    x: [0, 100, 200, ..., 900] (10 pts), y: same. Chunks (5, 5) => 2x2 chunks.
    Interpolate at (250, 250): x bracket (2,3), y bracket (2,3). All in chunk (0,0).
    """
    import numpy as np
    import xarray as xr

    nx, ny = 10, 10
    x_coords = np.array([i * 100.0 for i in range(nx)], dtype=np.float64)
    y_coords = np.array([i * 100.0 for i in range(ny)], dtype=np.float64)
    data = np.arange(nx * ny, dtype=np.float64).reshape(ny, nx)

    ds = xr.Dataset(
        data_vars={"value": (["y", "x"], data)},
        coords={"x": x_coords, "y": y_coords},
    )

    zarr_path = tmp_path / "interp_sparse_2d.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (5, 5),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # (250, 250) is between (200,300) in x and (200,300) in y. Indices (2,3) x (2,3). Chunk (0,0).
    target = pl.DataFrame({"x": [250.0], "y": [250.0], "label": ["mid"]})
    expr = interpolate_nd(["y", "x"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    assert idxs == {(0, 0)}


def test_interpolate_nd_sparse_2d_coords_bracket_spans_chunk_boundaries(tmp_path: Path) -> None:
    """2D sparse coords: interpolation point where brackets span chunk boundaries.

    x: [0, 100, ..., 900], y: same. Chunks (5, 5). Interpolate at (450, 450):
    x bracket (4,5), y bracket (4,5). Index 4 in chunk 0, index 5 in chunk 1 for each dim.
    Must select all 4 chunks: (0,0), (0,1), (1,0), (1,1).
    """
    import numpy as np
    import xarray as xr

    nx, ny = 10, 10
    x_coords = np.array([i * 100.0 for i in range(nx)], dtype=np.float64)
    y_coords = np.array([i * 100.0 for i in range(ny)], dtype=np.float64)
    data = np.arange(nx * ny, dtype=np.float64).reshape(ny, nx)

    ds = xr.Dataset(
        data_vars={"value": (["y", "x"], data)},
        coords={"x": x_coords, "y": y_coords},
    )

    zarr_path = tmp_path / "interp_sparse_2d_boundary.zarr"
    ds.to_zarr(
        zarr_path,
        zarr_format=3,
        encoding={
            "value": {
                "chunks": (5, 5),
                "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)],
            }
        },
    )

    # (450, 450) between (400,500) in both dims. Indices (4,5) x (4,5). Spans all 4 chunks.
    target = pl.DataFrame({"x": [450.0], "y": [450.0], "label": ["corner"]})
    expr = interpolate_nd(["y", "x"], ["value"], target)
    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks, variable="value")

    assert idxs == {(0, 0), (0, 1), (1, 0), (1, 1)}

