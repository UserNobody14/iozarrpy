from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

import rainbear
import rainbear._core as _core


def _write_1d_dataset(path: Path, *, n: int = 10_000, chunk: int = 1_000) -> str:
    coord = np.arange(n, dtype=np.int64)
    data = np.zeros(n, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={"v": ("mycoord", data)},
        coords={"mycoord": ("mycoord", coord)},
    )
    encoding = {
        "v": {"chunks": (chunk,)},
        "mycoord": {"chunks": (chunk,)},
    }
    ds.to_zarr(str(path), zarr_format=3, encoding=encoding)
    return str(path)


def test_selected_chunks_disjoint_or_ranges(tmp_path: Path) -> None:
    zarr_path = _write_1d_dataset(tmp_path / "one_d.zarr", n=10_000, chunk=1_000)

    expr = ((pl.col("mycoord") > 20) & (pl.col("mycoord") < 30)) | (
        (pl.col("mycoord") > 7000) & (pl.col("mycoord") < 7010)
    )

    chunks = rainbear.selected_chunks(zarr_path, expr, variables=["v"])  # type: ignore[attr-defined]
    idxs = sorted({tuple(c["indices"]) for c in chunks})

    # With chunk size 1000, 20..30 is in chunk 0 and 7000..7010 is in chunk 7.
    assert idxs == [(0,), (7,)]


def test_planner_coord_reads_are_sublinear(tmp_path: Path) -> None:
    zarr_path = _write_1d_dataset(tmp_path / "one_d_big.zarr", n=100_000, chunk=1_000)

    expr = ((pl.col("mycoord") > 20) & (pl.col("mycoord") < 30)) | (
        (pl.col("mycoord") > 70_000) & (pl.col("mycoord") < 70_010)
    )

    chunks, coord_reads = _core._selected_chunks_debug(zarr_path, expr, variables=["v"])
    idxs = sorted({tuple(c["indices"]) for c in chunks})
    assert idxs == [(0,), (70,)]

    # Heuristic: should be O(log N) coord reads, not O(#chunks).
    assert coord_reads < 500


def test_selected_chunks_index_only_dims(tmp_path: Path) -> None:
    # Store with dims (y, x) but *without* 1D coord arrays for y/x.
    zarr_path = str(tmp_path / "index_only_dims.zarr")
    _core._create_index_only_store(zarr_path)

    expr = (pl.col("y") == 0) & (pl.col("x") >= 0)
    chunks = rainbear.selected_chunks(zarr_path, expr, variables=["var"])  # type: ignore[attr-defined]

    idxs = sorted({tuple(c["indices"]) for c in chunks})
    # y==0 constrains to y-chunk 0; x unconstrained across 2 chunks.
    assert idxs == [(0, 0), (0, 1)]

