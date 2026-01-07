"""Tests for predicate pushdown expression variants (using pre-generated output datasets)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rainbear import _core


def _demo_store_path() -> str:
    p = Path(__file__).resolve().parent / "output-datasets" / "demo_store.zarr"
    assert p.exists(), f"expected pre-generated dataset at {p}"
    return str(p)


def _chunk_indices(chunks: list[object]) -> set[tuple[int, ...]]:
    # _core._selected_chunks_debug returns list of dicts like {"indices": [...], "origin": [...], "shape": [...]}
    out: set[tuple[int, ...]] = set()
    for d in chunks:
        idx = d["indices"]
        out.add(tuple(int(x) for x in idx))
    return out

# Failing
def test_is_between_pushdown_narrows_chunks() -> None:
    zarr_url = _demo_store_path()
    pred = pl.col("time").is_between(0, 1, closed="both")
    chunks, coord_reads = _core._selected_chunks_debug(zarr_url, pred, variables=["temp"])
    assert coord_reads >= 0
    assert _chunk_indices(chunks) == {(0, 0)}


# Failing
def test_is_in_pushdown_narrows_chunks() -> None:
    zarr_url = _demo_store_path()
    pred = pl.col("time").is_in([0, 2])
    chunks, _coord_reads = _core._selected_chunks_debug(zarr_url, pred, variables=["temp"])
    assert _chunk_indices(chunks) == {(0, 0)}

# Failing
def test_wrappers_alias_cast_preserve_pushdown() -> None:
    zarr_url = _demo_store_path()
    pred = pl.col("time").eq(0).alias("pushed").cast(pl.Boolean)
    chunks, _coord_reads = _core._selected_chunks_debug(zarr_url, pred, variables=["temp"])
    assert _chunk_indices(chunks) == {(0, 0)}

# Failing
def test_not_constant_folded() -> None:
    zarr_url = _demo_store_path()

    chunks, _ = _core._selected_chunks_debug(zarr_url, ~pl.lit(True), variables=["temp"])
    assert _chunk_indices(chunks) == set()

    chunks, _ = _core._selected_chunks_debug(zarr_url, ~pl.lit(False), variables=["temp"])
    assert _chunk_indices(chunks) == {(0, 0), (1, 0)}

# Failing
def test_is_null_is_not_null_constant_folded() -> None:
    zarr_url = _demo_store_path()

    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.lit(None).is_null(), variables=["temp"])
    assert _chunk_indices(chunks) == {(0, 0), (1, 0)}

    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.lit(None).is_not_null(), variables=["temp"])
    assert _chunk_indices(chunks) == set()


# Failing
def test_and_or_with_unsupported_side() -> None:
    zarr_url = _demo_store_path()

    # Ternary expressions are not supported by the Rust chunk planner. They should not break AND pushdown.
    ternary = pl.when(pl.lit(True)).then(pl.lit(True)).otherwise(pl.lit(False))

    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.col("time").eq(0) & ternary, variables=["temp"])
    assert _chunk_indices(chunks) == {(0, 0)}

    # For OR, any unknown side must conservatively become "all chunks" to avoid false negatives.
    chunks, _ = _core._selected_chunks_debug(zarr_url, pl.col("time").eq(0) | ternary, variables=["temp"])
    assert _chunk_indices(chunks) == {(0, 0), (1, 0)}

