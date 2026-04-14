from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import xarray as xr

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import SelectedChunksDebugReturn

def _chunk_index_list(chunks_result: SelectedChunksDebugReturn, variable: str) -> list[tuple[int, ...]]:
    """Return the raw (possibly duplicate) list of chunk indices for a variable."""
    for grid in chunks_result["grids"]:
        if variable in grid["variables"]:
            return [tuple(int(x) for x in c["indices"]) for c in grid["chunks"]]
    raise ValueError(f"No grid found for variable '{variable}' in {chunks_result}")


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

    chunks = ZarrBackend.from_url(zarr_path).selected_chunks_debug(expr)  # type: ignore[attr-defined]
    # Find a grid that includes "mycoord"
    for grid in chunks["grids"]:
        if "v" in grid["variables"]:
            idxs = sorted({tuple(c["indices"]) for c in grid["chunks"]})
            break
    else:
        raise ValueError(f"No grid found for variable 'mycoord' in {chunks}")

    # With chunk size 1000, 20..30 is in chunk 0 and 7000..7010 is in chunk 7.
    assert idxs == [(0,), (7,)]


def test_planner_coord_reads_are_sublinear(tmp_path: Path) -> None:
    zarr_path = _write_1d_dataset(tmp_path / "one_d_big.zarr", n=100_000, chunk=1_000)

    expr = ((pl.col("mycoord") > 20) & (pl.col("mycoord") < 30)) | (
        (pl.col("mycoord") > 70_000) & (pl.col("mycoord") < 70_010)
    )

    chunks = ZarrBackend.from_url(zarr_path).selected_chunks_debug(expr)
    coord_reads = chunks["coord_reads"]
    # Find a grid that includes "mycoord"
    for grid in chunks["grids"]:
        if "v" in grid["variables"]:
            idxs = sorted({tuple(c["indices"]) for c in grid["chunks"]})
            break
    else:
        raise ValueError(f"No grid found for variable 'mycoord' in {chunks}")
    assert idxs == [(0,), (70,)]

    # Heuristic: should be O(log N) coord reads, not O(#chunks).
    assert coord_reads < 500


def test_selected_chunks_index_only_dims(baseline_datasets: dict[str, str]) -> None:
    # Store with dims (y, x) but *without* 1D coord arrays for y/x.
    zarr_path = baseline_datasets["index_only_dims"]

    expr = (pl.col("y") == 0) & (pl.col("x") >= 0)
    chunks = ZarrBackend.from_url(zarr_path).selected_chunks_debug(expr)  # type: ignore[attr-defined]
    # Find a grid that includes "var"
    for grid in chunks["grids"]:
        if "var" in grid["variables"]:
            idxs = sorted({tuple(c["indices"]) for c in grid["chunks"]})
            break
    else:
        raise ValueError(f"No grid found for variable 'mycoord' in {chunks}")

    # y==0 constrains to y-chunk 0; x unconstrained across 2 chunks.
    assert idxs == [(0, 0), (0, 1)]


def test_no_duplicate_chunks_overlapping_or_subsets(tmp_path: Path) -> None:
    """OR conditions with overlapping ranges must not produce duplicate chunks in debug output.

    When two OR branches both cover the same chunk, `selected_chunks_debug` must return
    that chunk exactly once.  Before the fix, the non-sharded path iterated over each
    subset independently without deduplication.
    """
    n, chunk_size = 20, 10
    coord = np.arange(n, dtype=np.int64)
    data = np.zeros(n, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={"v": ("x", data)},
        coords={"x": ("x", coord)},
    )
    zarr_path = tmp_path / "overlap.zarr"
    ds.to_zarr(str(zarr_path), zarr_format=3, encoding={"v": {"chunks": (chunk_size,)}, "x": {"chunks": (chunk_size,)}})

    # Both branches resolve to chunk 0 (x in [0, 10)).
    expr = ((pl.col("x") >= 0) & (pl.col("x") <= 3)) | ((pl.col("x") >= 5) & (pl.col("x") <= 8))

    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)
    raw_list = _chunk_index_list(chunks, "v")

    # No duplicates: the raw list length must equal the unique-set length.
    assert len(raw_list) == len(set(raw_list)), (
        f"Duplicate chunk indices in debug output: {raw_list}"
    )
    # Exactly one chunk selected (chunk 0).
    assert set(raw_list) == {(0,)}


def test_no_duplicate_chunks_v2_multi_variable_or_query(tmp_path: Path) -> None:
    """v2 dataset with many variables + overlapping OR query must not produce duplicates.

    This reproduces the reported pattern: a Zarr v2 dataset, many variables sharing the
    same chunk grid, and a query that generates multiple overlapping subsets (e.g. an
    OR of two conditions that both fall in the same chunk).
    """
    n, chunk_size = 30, 10
    x = np.arange(n, dtype=np.float64)
    var_names = [f"var_{i}" for i in range(8)]
    data_vars = {name: ("x", np.zeros(n, dtype=np.float32)) for name in var_names}
    ds = xr.Dataset(data_vars=data_vars, coords={"x": ("x", x)})

    encoding = {name: {"chunks": (chunk_size,)} for name in var_names}
    encoding["x"] = {"chunks": (chunk_size,)}

    zarr_path = tmp_path / "v2_multi_var.zarr"
    ds.to_zarr(str(zarr_path), zarr_format=2, encoding=encoding)

    # Two overlapping branches: both fall within chunk 0 (x in [0, 10)).
    expr = ((pl.col("x") >= 1.0) & (pl.col("x") <= 4.0)) | ((pl.col("x") >= 3.0) & (pl.col("x") <= 7.0))

    chunks = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)

    for grid in chunks["grids"]:
        raw_list = [tuple(int(v) for v in c["indices"]) for c in grid["chunks"]]
        assert len(raw_list) == len(set(raw_list)), (
            f"Duplicate chunk indices for grid {grid['variables']}: {raw_list}"
        )

    # All variables in the same grid; only chunk 0 should be selected.
    raw_list = _chunk_index_list(chunks, "var_0")
    assert set(raw_list) == {(0,)}
