"""More chunk-planner integration tests.

Key rule: these tests must not be able to pass if the planner just returns “all chunks”.
So every test asserts strict narrowing vs the full chunk grid and usually exact chunk sets.
"""

from __future__ import annotations

from datetime import datetime, timedelta
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
    raise ValueError(f"No grid found for variable '2m_temperature' in {chunks}")


def _grid_meta(zarr_key: str) -> tuple[int, int]:
    """Return (all_total, per_x_chunk) for stable assertions on baseline datasets."""
    if zarr_key == "grid_chunked":
        all_total = 3 * 5 * 4 * 4  # (time, lead_time, y, x) with chunks=(1,2,100,100)
        per_x = 3 * 5 * 4 * 1
        return all_total, per_x
    if zarr_key == "grid_sharded":
        # With shards=(1,4,200,200), zarrs treats the shard grid as the "chunk grid".
        all_total = 3 * 3 * 2 * 2
        per_x = 3 * 3 * 2 * 1
        return all_total, per_x
    raise AssertionError(f"unexpected dataset key: {zarr_key}")


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_xor_narrows_chunks(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, per_x = _grid_meta(zarr_key)

    # Pick indices that don't accidentally cover *all* x-chunks.
    rhs_x = 200 if zarr_key == "grid_chunked" else 50
    pred = (pl.col("x") == 0) ^ (pl.col("x") == rhs_x)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)

    expected_count = 2 * per_x if zarr_key == "grid_chunked" else per_x
    assert len(idxs) == expected_count
    assert len(idxs) < all_total
    # x is last dim here (time, lead_time, y, x)
    assert {t[3] for t in idxs} == ({0, 2} if zarr_key == "grid_chunked" else {0})


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_any_all_horizontal_narrow(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, per_x = _grid_meta(zarr_key)

    rhs_x = 200 if zarr_key == "grid_chunked" else 50
    any_pred = pl.any_horizontal([pl.col("x") == 0, pl.col("x") == rhs_x])
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( any_pred)
    idxs = _chunk_indices(chunks)
    expected_count = 2 * per_x if zarr_key == "grid_chunked" else per_x
    assert len(idxs) == expected_count
    assert len(idxs) < all_total

    all_pred = pl.all_horizontal([pl.col("x") >= 0, pl.col("x") <= 1])
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( all_pred)
    idxs = _chunk_indices(chunks)
    assert len(idxs) == per_x
    assert len(idxs) < all_total
    assert {t[3] for t in idxs} == {0}


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_over_window_preserves_pushdown(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, per_x = _grid_meta(zarr_key)

    pred = (pl.col("x") == 0).over("y")
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)

    assert len(idxs) == per_x
    assert len(idxs) < all_total
    assert {t[3] for t in idxs} == {0}


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_datetime_and_duration_constraints_narrow(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, _per_x = _grid_meta(zarr_key)

    # Pick first time in generator: datetime(2024,1,1) (see tests/zarr_generators.py)
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    pred_time = (pl.col("time") >= t0) & (pl.col("time") <= t0)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred_time)
    idxs = _chunk_indices(chunks)

    # Selecting a single time chunk should be a strict reduction vs all.
    assert 0 < len(idxs) < all_total

    # Lead time is a duration coord in the generated datasets.
    lt0 = timedelta(hours=0)
    pred_lt = (pl.col("lead_time") >= lt0) & (pl.col("lead_time") <= lt0)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred_lt)
    idxs2 = _chunk_indices(chunks)
    assert 0 < len(idxs2) < all_total


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_null_nan_terms_do_not_break_dim_narrowing(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, per_x = _grid_meta(zarr_key)

    pred = (pl.col("x") == 0) & pl.col("2m_temperature").is_nan()
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)
    assert len(idxs) == per_x
    assert len(idxs) < all_total

    pred = (pl.col("x") == 0) & pl.col("2m_temperature").is_null()
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)
    assert len(idxs) == per_x
    assert len(idxs) < all_total


def test_negation_cuts_hole_in_selected_cube(baseline_datasets: dict[str, str]) -> None:
    # This should cut away chunk (0,0) while keeping the rest.
    zarr_url = baseline_datasets["orography_chunked_10x10"]

    outer = (pl.col("y") >= 0) & (pl.col("y") <= 15) & (pl.col("x") >= 0) & (pl.col("x") <= 19)
    hole = (pl.col("y") >= 0) & (pl.col("y") <= 9) & (pl.col("x") >= 0) & (pl.col("x") <= 9)
    pred = outer & ~hole

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = sorted(_chunk_indices(chunks, variable="geopotential_height"))

    # Grid is 2x2 with chunks (10,10). Removing the (0,0) block leaves 3 blocks.
    assert idxs == [(0, 1), (1, 0), (1, 1)]

