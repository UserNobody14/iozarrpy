"""Planner tests for `interpolate_nd` FFI plugin expressions.

These tests only validate *chunk planning* (not plugin evaluation), and are written so they
cannot pass by returning “all chunks”.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from interpolars import interpolate_nd

from rainbear import _core


def _chunk_indices(chunks: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    return {tuple(int(x) for x in d["indices"]) for d in chunks}


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
    chunks, _ = _core._selected_chunks_debug(zarr_url, expr, variables=["geopotential_height"])
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
    chunks, _ = _core._selected_chunks_debug(zarr_url, expr, variables=["geopotential_height"])
    idxs = _chunk_indices(chunks)

    # y stays in chunk 0, x straddles boundary at 10 => chunks (0,0) and (0,1).
    assert idxs == {(0, 0), (0, 1)}

