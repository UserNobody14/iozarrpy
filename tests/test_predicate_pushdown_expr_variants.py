"""Tests for predicate pushdown expression variants (using pre-generated output datasets)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import SelectedChunksDebugReturn


def _chunk_indices(chunks: SelectedChunksDebugReturn, variable: str = "2m_temperature") -> set[tuple[int, ...]]:
    # Find a grid that includes the variable
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")


def _assert_grid_x_selection(
    idxs: set[tuple[int, ...]],
    *,
    expected_x_chunks: set[int],
    expected_total: int,
) -> None:
    assert idxs or expected_total == 0
    for idx in idxs:
        assert len(idx) == 4, f"Expected 4 dimensions, got {len(idx)}: {idx}"
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
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    coord_reads = chunks["coord_reads"]
    assert coord_reads >= 0
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)


def test_is_in_pushdown_narrows_chunks(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, per_x = _grid_dataset
    pred = (pl.col("x") == 0) | (pl.col("x") == 200)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)
    expected_x = {0, 2} if per_x == 3 * 5 * 4 * 1 else {0, 1}
    _assert_grid_x_selection(idxs, expected_x_chunks=expected_x, expected_total=2 * per_x)


def test_wrappers_alias_cast_preserve_pushdown(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, per_x = _grid_dataset
    pred = pl.col("x").eq(0).alias("pushed").cast(pl.Boolean)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)


def test_literal_true_false_are_constant_folded(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, all_total, _per_x = _grid_dataset

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.lit(False))
    with pytest.raises(ValueError):
        _chunk_indices(chunks)

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.lit(True))
    idxs = _chunk_indices(chunks)
    assert len(idxs) == all_total


def test_literal_null_keeps_nothing(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, _per_x = _grid_dataset
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.lit(None))
    with pytest.raises(ValueError):
        _chunk_indices(chunks)


def test_and_or_with_complex_side(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, all_total, per_x = _grid_dataset

    # More complex expressions should not break AND pushdown.
    ternary = pl.when(pl.lit(True)).then(pl.lit(True)).otherwise(pl.lit(True))

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.col("x").eq(0) & ternary
    )
    idxs = _chunk_indices(chunks, variable="total_precipitation")
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)

    # For OR, any unknown side must conservatively become "all chunks" to avoid false negatives.
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.col("x").eq(0) | ternary
    )
    idxs = _chunk_indices(chunks, variable="total_precipitation")
    assert len(idxs) == all_total



def test_filter_on_alias_preserves_pushdown(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, _all_total, per_x = _grid_dataset

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.col("x").eq(0).alias("pushed").cast(pl.Boolean)
    )
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)


def test_filter_on_alias_complex_expression_preserves_pushdown(_grid_dataset: tuple[str, int, int]) -> None:
    zarr_url, all_total, per_x = _grid_dataset

    # More complex expressions should not break AND pushdown.
    flt = pl.col("x").alias("filtertest").eq(0)
    flt = flt.alias("pushed").cast(pl.Boolean)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( flt
    )
    
    idxs = _chunk_indices(chunks)
    _assert_grid_x_selection(idxs, expected_x_chunks={0}, expected_total=per_x)

