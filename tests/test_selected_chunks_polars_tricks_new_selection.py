"""More Polars expression variants for `selected_chunks` (planner smoke + stability).

These are intentionally focused on tricky Expr shapes that can appear in real code:
- boolean functions: is_between, is_in, is_null, not
- wrappers: alias, cast, keep_name (when available)
- ternary: when/then/otherwise
"""

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

def _grid_expected(zarr_key: str) -> tuple[int, int, set[int]]:
    """Return (all_total, per_x, expected_x_chunks_for_0_and_200)."""
    if zarr_key == "grid_chunked":
        all_total = 3 * 5 * 4 * 4  # chunks=(1,2,100,100)
        per_x = 3 * 5 * 4 * 1
        return (all_total, per_x, {0, 2})
    if zarr_key == "grid_sharded":
        # With shards=(1,4,200,200), zarrs treats shard grid as chunk grid.
        all_total = 3 * 3 * 2 * 2
        per_x = 3 * 3 * 2 * 1
        return (all_total, per_x, {0, 1})
    raise AssertionError(f"unexpected dataset key: {zarr_key}")


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_selected_chunks_is_in_method(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, per_x, expected_x = _grid_expected(zarr_key)

    pred = pl.col("x").is_in([0, 200])
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)

    idxs = _chunk_indices(chunks)

    assert len(idxs) == 2 * per_x
    assert {t[3] for t in idxs} == expected_x
    assert len(idxs) <= all_total


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_selected_chunks_is_between_method(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    _all_total, per_x, _expected_x = _grid_expected(zarr_key)

    pred = pl.col("x").is_between(0, 1)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)

    assert len(idxs) == per_x
    assert {t[3] for t in idxs} == {0}


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_selected_chunks_not_is_conservative(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, _per_x, _expected_x = _grid_expected(zarr_key)

    pred = ~(pl.col("x") == 0)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)

    # We don't represent complements; must be conservative (all chunks).
    assert len(idxs) == all_total


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_selected_chunks_is_null_is_not_null_do_not_crash(
    baseline_datasets: dict[str, str], zarr_key: str
) -> None:
    zarr_url = baseline_datasets[zarr_key]
    all_total, _per_x, _expected_x = _grid_expected(zarr_key)

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.col("x").is_null()
    )
    assert len(_chunk_indices(chunks)) == all_total

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pl.col("x").is_not_null()
    )
    assert len(_chunk_indices(chunks)) == all_total


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_selected_chunks_ternary_when_then_otherwise(baseline_datasets: dict[str, str], zarr_key: str) -> None:
    zarr_url = baseline_datasets[zarr_key]
    _all_total, per_x, _expected_x = _grid_expected(zarr_key)

    pred = pl.when(pl.col("x") == 0).then(pl.lit(True)).otherwise(pl.lit(False))
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)

    assert len(idxs) == per_x
    assert {t[3] for t in idxs} == {0}


@pytest.mark.parametrize("zarr_key", ["grid_chunked", "grid_sharded"])
def test_selected_chunks_wrappers_alias_cast_keep_name(
    baseline_datasets: dict[str, str], zarr_key: str
) -> None:
    zarr_url = baseline_datasets[zarr_key]
    _all_total, per_x, _expected_x = _grid_expected(zarr_key)

    pred: pl.Expr = pl.col("x").eq(0).alias("pushed").cast(pl.Boolean)
    # keep_name exists in Polars but not all versions/paths produce Expr::KeepName; this test
    # is still useful as a “should not break” wrapper chain.
    if hasattr(pred, "keep_name"):
        pred = pred.keep_name()  # type: ignore[union-attr]

    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = _chunk_indices(chunks)

    assert len(idxs) == per_x
    assert {t[3] for t in idxs} == {0}


def test_selected_chunks_coord_var_yx_not_overselected(baseline_datasets: dict[str, str]) -> None:
    # Orography dataset includes latitude/longitude as (y, x) variables.
    zarr_url = baseline_datasets["orography_chunked_10x10"]

    pred = (pl.col("y") >= 0) & (pl.col("y") <= 8) & (pl.col("x") >= 0) & (pl.col("x") <= 8)
    chunks = ZarrBackend.from_url(zarr_url).selected_chunks_debug( pred)
    idxs = sorted(_chunk_indices(chunks, variable="geopotential_height"))

    # With chunks (10,10) and ny=16,nx=20, y<=8 and x<=8 hits only chunk (0,0).
    assert idxs == [(0, 0)]

