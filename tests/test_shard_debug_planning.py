"""Tests for `selected_chunks_debug` shard metadata on chunk entries.

Non-sharded stores expose an empty `shards` list. Sharded Zarr v3 arrays attach one
`ShardInfo` per inner chunk (indices / origin / shape in the outer shard grid), while
`indices` / `origin` / `shape` on the chunk remain the inner chunk grid—the same
layout as the non-sharded planner for equivalent chunk encodings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import GridInfo, SelectedChunksDebugReturn


def _grid_for_variable(debug: SelectedChunksDebugReturn, variable: str) -> GridInfo:
    for grid in debug["grids"]:
        if variable in grid["variables"]:
            return grid
    raise ValueError(f"No grid for variable {variable!r}")


def _chunk_index_set(debug: SelectedChunksDebugReturn, variable: str) -> set[tuple[int, ...]]:
    grid = _grid_for_variable(debug, variable)
    return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}


def test_non_sharded_chunks_have_empty_shards(baseline_datasets: dict[str, str]) -> None:
    pred = pl.col("x").is_between(0, 1)
    debug = ZarrBackend.from_url(baseline_datasets["grid_chunked"]).selected_chunks_debug(pred)
    grid = _grid_for_variable(debug, "2m_temperature")
    assert len(grid["chunks"]) > 0
    for c in grid["chunks"]:
        assert c["shards"] == []


def test_sharded_grid_matches_chunked_inner_indices(baseline_datasets: dict[str, str]) -> None:
    pred = (pl.col("x") >= 0) & (pl.col("x") <= 1)
    chunked = ZarrBackend.from_url(baseline_datasets["grid_chunked"]).selected_chunks_debug(pred)
    sharded = ZarrBackend.from_url(baseline_datasets["grid_sharded"]).selected_chunks_debug(pred)
    assert _chunk_index_set(chunked, "2m_temperature") == _chunk_index_set(
        sharded, "2m_temperature"
    )


def test_sharded_chunks_include_shard_info_covering_inner_extent(
    baseline_datasets: dict[str, str],
) -> None:
    pred = pl.lit(True)
    debug = ZarrBackend.from_url(baseline_datasets["grid_sharded"]).selected_chunks_debug(pred)
    grid = _grid_for_variable(debug, "2m_temperature")
    assert len(grid["chunks"]) == 3 * 5 * 4 * 4
    for c in grid["chunks"]:
        assert len(c["shards"]) == 1
        s = c["shards"][0]
        assert len(s["indices"]) == len(grid["dims"])
        assert len(s["origin"]) == len(grid["dims"])
        assert len(s["shape"]) == len(grid["dims"])
        for d in range(len(grid["dims"])):
            assert c["origin"][d] >= s["origin"][d]
            assert c["origin"][d] + c["shape"][d] <= s["origin"][d] + s["shape"][d]


def test_orography_sharded_inner_chunks_share_fewer_unique_shards(
    baseline_datasets: dict[str, str],
) -> None:
    """chunks=(4,4), shards=(8,8): up to four inner chunks per shard; unique shards < chunks."""
    debug = ZarrBackend.from_url(baseline_datasets["orography_sharded_small"]).selected_chunks_debug(
        pl.lit(True)
    )
    grid = _grid_for_variable(debug, "geopotential_height")
    assert len(grid["chunks"]) == 12  # 4 x 3 inner chunks for 16x12 with chunk 4x4
    shard_index_tuples: list[tuple[int, ...]] = []
    for c in grid["chunks"]:
        assert len(c["shards"]) == 1
        shard_index_tuples.append(tuple(int(x) for x in c["shards"][0]["indices"]))
    assert len(set(shard_index_tuples)) < len(grid["chunks"])


def test_orography_sharded_shard_bounds_enclose_chunk(baseline_datasets: dict[str, str]) -> None:
    debug = ZarrBackend.from_url(baseline_datasets["orography_sharded_small"]).selected_chunks_debug(
        pl.lit(True)
    )
    grid = _grid_for_variable(debug, "geopotential_height")
    for c in grid["chunks"]:
        s = c["shards"][0]
        for d in range(2):
            assert c["origin"][d] >= s["origin"][d]
            assert c["origin"][d] + c["shape"][d] <= s["origin"][d] + s["shape"][d]
