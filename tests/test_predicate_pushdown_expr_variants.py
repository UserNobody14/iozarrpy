"""Tests for predicate pushdown expression variants (using pre-generated output datasets)."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from rainbear import _core


def _chunk_indices(chunks: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    # _core._selected_chunks_debug returns list of dicts like {"indices": [...], "origin": [...], "shape": [...]}
    out: set[tuple[int, ...]] = set()
    for d in chunks:
        idx = d["indices"]
        out.add(tuple(int(x) for x in idx))
    return out


def _assert_grid_x_selection(
    idxs: set[tuple[int, ...]],
    *,
    expected_x_chunks: set[int],
    expected_total: int,
) -> None:
    assert idxs or expected_total == 0
    assert {t[3] for t in idxs} == expected_x_chunks
    assert len(idxs) == expected_total


@pytest.fixture(params=["grid_chunked", "grid_sharded"])
def _grid_dataset(
    baseline_datasets: dict[str, str], request: pytest.FixtureRequest
) -> tuple[str, int, int]:
    """Return (zarr_url, all_chunks_total, chunks_per_x_chunk) for stable assertions."""
    key = str(request.param)
    zarr_url = baseline_datasets[key]
    if key == "grid_chunked":
        all_total = 3 * 5 * 4 * 4  # (time, lead_time, y, x) chunk grid with chunks=(1,2,100,100)
        per_x = 3 * 5 * 4 * 1
        return (zarr_url, all_total, per_x)
    if key == "grid_sharded":
        # With shards=(1,4,200,200), zarrs treats the shard grid as the "chunk grid" here.
        all_total = 3 * 3 * 2 * 2
        per_x = 3 * 3 * 2 * 1
        return (zarr_url, all_total, per_x)
    raise AssertionError(f"unexpected dataset key: {key}")


def test_is_between_pushdown_narrows_chunks(_grid_dataset: tuple[str, int, int]) -> None:
    # Uses a baseline dataset with explicit chunking for stable chunk-grid expectations.
    zarr_url, _all_total, per_x = _grid_dataset
    pred = (pl.col("x") >= 0) & (pl.col("x") <= 1)
    chunks, coord_reads = _core._selected_chunks_debug(zarr_url, pred, variables=["2m_temperature"])
    assert coord_reads >= 0
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)


def test_is_in_pushdown_narrows_chunks(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, per_x = _grid_dataset
    pred = (pl.col("x") == 0) | (pl.col("x") == 200)
    chunks, _coord_reads = _core._selected_chunks_debug(zarr_url, pred, variables=["2m_temperature"])
    idxs = _chunk_indices(chunks)
    expected_x = {0, 2} if per_x == 3 * 5 * 4 * 1 else {0, 1}
    _assert_grid_x_selection(idxs, expected_x_chunks=expected_x, expected_total=2 * per_x)


def test_wrappers_alias_cast_preserve_pushdown(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, per_x = _grid_dataset
    pred = pl.col("x").eq(0).alias("pushed").cast(pl.Boolean)
    chunks, _coord_reads = _core._selected_chunks_debug(zarr_url, pred, variables=["2m_temperature"])
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)


def test_literal_true_false_are_constant_folded(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, all_total, _per_x = _grid_dataset

    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.lit(False), variables=["2m_temperature"])
    assert _chunk_indices(chunks) == set()

    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.lit(True), variables=["2m_temperature"])
    idxs = _chunk_indices(chunks)
    assert len(idxs) == all_total


def test_literal_null_keeps_nothing(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, _per_x = _grid_dataset
    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.lit(None), variables=["2m_temperature"])
    assert _chunk_indices(chunks) == set()


def test_and_or_with_complex_side(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, all_total, per_x = _grid_dataset

    # More complex expressions should not break AND pushdown.
    ternary = pl.when(pl.lit(True)).then(pl.lit(True)).otherwise(pl.lit(True))

    chunks, _ = _core._selected_chunks_debug(
        zarr_url, pl.col("x").eq(0) & ternary, variables=["2m_temperature"]
    )
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)

    # For OR, any unknown side must conservatively become "all chunks" to avoid false negatives.
    chunks, _ = _core._selected_chunks_debug(
        zarr_url, pl.col("x").eq(0) | ternary, variables=["2m_temperature"]
    )
    idxs = _chunk_indices(chunks)
    assert len(idxs) == all_total

