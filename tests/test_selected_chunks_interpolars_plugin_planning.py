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

