"""Tests that Expr::Filter correctly constrains chunk planning.

Regression tests for a bug where pl.col(columns).filter(pred) was using
union instead of intersect in the chunk planner, causing the filter predicate
to be completely ignored. This resulted in the planner selecting ALL chunks
in the dataset rather than just the ones matching the predicate.

The bug caused massive chunk count explosions (e.g. 9 million chunks when
only 1 was needed) and triggered max_chunks_to_read safety limits.
"""

from __future__ import annotations

import asyncio
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import xarray as xr

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from conftest import ComprehensiveDatasetInfo

    from rainbear._core import SelectedChunksDebugReturn


# =============================================================================
# Helper functions
# =============================================================================


def _total_chunks(debug_info: SelectedChunksDebugReturn) -> int:
    """Count total chunks across all grids from selected_chunks_debug output."""
    return sum(len(grid["chunks"]) for grid in debug_info["grids"])


def _per_var_chunks(debug_info: SelectedChunksDebugReturn) -> dict[str, int]:
    """Get per-variable chunk counts from selected_chunks_debug output."""
    per_var: dict[str, int] = {}
    for grid in debug_info["grids"]:
        for var in grid["variables"]:
            per_var[var] = len(grid["chunks"])
    return per_var


def _all_vars(debug_info: dict) -> set[str]:
    """Get all variable names from selected_chunks_debug output."""
    return set(chain.from_iterable(
        grid["variables"] for grid in debug_info["grids"]
    ))


# =============================================================================
# Core regression test: filter expression chunk counts
# =============================================================================


class TestFilterExprChunkPlanning:
    """Test that pl.col(columns).filter(pred) correctly limits chunks."""

    def test_filter_expr_produces_same_chunks_as_bare_predicate(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """The chunk plan for pl.col(cols).filter(pred) should match the
        bare predicate's plan (both should use the predicate for pruning)."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        pred = (pl.col("a") >= 10) & (pl.col("a") < 30)
        cols = ["a", "b", "c", "data"]

        # Bare predicate: directly constrains chunks
        bare_plan = backend.selected_chunks_debug(pred)
        bare_chunks = _per_var_chunks(bare_plan)

        # Filtered expression: should produce the SAME chunk plan
        filter_expr = pl.col(cols).filter(pred)
        filter_plan = backend.selected_chunks_debug(filter_expr)
        filter_chunks = _per_var_chunks(filter_plan)

        # The filter expression should NOT select more chunks than the bare predicate
        assert filter_chunks["data"] <= bare_chunks["data"], (
            f"Filter expression selected {filter_chunks['data']} chunks for 'data', "
            f"but bare predicate only selected {bare_chunks['data']}. "
            f"The filter predicate is being ignored in chunk planning!"
        )

    def test_filter_expr_does_not_select_all_chunks(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """A tight filter should NOT select all chunks in the dataset."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        # Select a narrow range: only 2 out of 7 chunks on dimension 'a'
        pred = (pl.col("a") >= 10) & (pl.col("a") < 30)
        cols = ["a", "b", "c", "data"]
        filter_expr = pl.col(cols).filter(pred)

        plan = backend.selected_chunks_debug(filter_expr)
        total = _total_chunks(plan)
        max_total = comprehensive_3d_dataset.total_chunks

        # The filter should prune significantly. Without the fix, total == max_total.
        assert total < max_total, (
            f"Filter expression selected ALL {total} chunks "
            f"(total in dataset = {max_total}). "
            f"Predicate pushdown is not working for Expr::Filter!"
        )

    def test_filter_expr_chunk_count_bounded(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """Verify specific chunk count bounds for a known query."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        # Exact equality on 'a': should hit at most 1 chunk on that dimension
        # a=15 is in chunk index 1 (chunk_size=10, so chunk 1 covers [10,20))
        pred = pl.col("a") == 15
        cols = ["a", "b", "c", "data"]
        filter_expr = pl.col(cols).filter(pred)

        plan = backend.selected_chunks_debug(filter_expr)
        chunks = _per_var_chunks(plan)

        # For data (7x5x3=105 total chunks):
        # a=15 hits 1 chunk on dim a, b unconstrained=5, c unconstrained=3
        # Expected: 1 * 5 * 3 = 15 chunks
        assert "data" in chunks
        assert chunks["data"] <= 15, (
            f"Expected at most 15 chunks for data with a==15, got {chunks['data']}"
        )

    def test_point_query_filter_expr(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """A point query (all dims constrained) should select very few chunks."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        pred = (
            (pl.col("a") == 15)
            & (pl.col("b") == 25)
            & (pl.col("c") == 10)
        )
        cols = ["a", "b", "c", "data"]
        filter_expr = pl.col(cols).filter(pred)

        plan = backend.selected_chunks_debug(filter_expr)
        total = _total_chunks(plan)

        # A point query should select ~1 chunk per grid group.
        # With the bug, it selected ALL 105 chunks.
        assert total <= 4, (
            f"Point query selected {total} chunks, expected at most ~1-4. "
            f"This suggests the filter predicate is being ignored."
        )


# =============================================================================
# max_chunks_to_read safety limit tests
# =============================================================================


class TestMaxChunksToReadWithFilter:
    """Test that max_chunks_to_read works correctly with filter expressions."""

    def test_filter_expr_respects_max_chunks_limit(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """A filtered scan with max_chunks_to_read should not hit the limit
        when the filter is tight enough."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        # Point query: should need ~1 chunk
        pred = (
            (pl.col("a") == 15)
            & (pl.col("b") == 25)
            & (pl.col("c") == 10)
        )
        cols = ["a", "b", "c", "data"]

        async def _run() -> pl.DataFrame:
            df = await backend.scan_zarr_async(
                pl.col(cols).filter(pred),
                max_chunks_to_read=10,  # Should be plenty for a point query
            )
            return df

        # This should NOT raise RuntimeError about max_chunks_to_read.
        # Before the fix, this would try to read 105 chunks and fail.
        df = asyncio.run(_run())
        assert df.height >= 0  # May be 0 if no exact match, but should not raise

    def test_tight_filter_does_not_trigger_chunk_limit(self, tmp_path: Path) -> None:
        """A tight filter on a larger dataset should stay within chunk limits."""
        # Create a dataset with many chunks
        n = 200
        chunk_size = 10  # 20 chunks per dimension, 400 total (2D)
        coord = np.arange(n, dtype=np.int64)
        data = np.random.default_rng(42).standard_normal((n, n)).astype(np.float32)
        ds = xr.Dataset(
            data_vars={"temp": (["y", "x"], data)},
            coords={
                "y": ("y", coord),
                "x": ("x", coord),
            },
        )
        zarr_path = str(tmp_path / "large_2d.zarr")
        ds.to_zarr(zarr_path, zarr_format=3, encoding={
            "temp": {"chunks": (chunk_size, chunk_size)},
        })

        backend = ZarrBackend.from_url(zarr_path)

        # Point query: y==50, x==50 → should hit 1 chunk
        pred = (pl.col("y") == 50) & (pl.col("x") == 50)
        cols = ["y", "x", "temp"]

        # With the bug, this would try to read 400 chunks and fail
        async def _run() -> pl.DataFrame:
            df = await backend.scan_zarr_async(
                pl.col(cols).filter(pred),
                max_chunks_to_read=5,
            )
            return df

        df = asyncio.run(_run())
        assert df.height >= 0

    def test_unfiltered_col_hits_chunk_limit(self, tmp_path: Path) -> None:
        """Selecting all columns WITHOUT a filter should correctly hit the limit."""
        n = 200
        chunk_size = 10
        coord = np.arange(n, dtype=np.int64)
        data = np.random.default_rng(42).standard_normal((n, n)).astype(np.float32)
        ds = xr.Dataset(
            data_vars={"temp": (["y", "x"], data)},
            coords={
                "y": ("y", coord),
                "x": ("x", coord),
            },
        )
        zarr_path = str(tmp_path / "large_2d_nofilt.zarr")
        ds.to_zarr(zarr_path, zarr_format=3, encoding={
            "temp": {"chunks": (chunk_size, chunk_size)},
        })

        backend = ZarrBackend.from_url(zarr_path)

        # No filter → should select ALL 400 chunks → exceed limit
        async def _run() -> pl.DataFrame:
            df = await backend.scan_zarr_async(
                pl.col(["y", "x", "temp"]),
                max_chunks_to_read=5,
            )
            return df

        with pytest.raises(RuntimeError, match="max_chunks_to_read exceeded"):
            asyncio.run(_run())


# =============================================================================
# Edge cases for filter expression planning
# =============================================================================


class TestFilterExprEdgeCases:
    """Edge cases for filter expression chunk planning."""

    def test_filter_with_non_dimension_predicate(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """A filter on a non-dimension column should still plan conservatively
        (can't prune on non-dimension predicates, but shouldn't crash)."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        # 'data' is not a dimension — filter can't prune chunks
        pred = pl.col("data") > 0.5
        cols = ["a", "b", "c", "data"]
        filter_expr = pl.col(cols).filter(pred)

        plan = backend.selected_chunks_debug(filter_expr)
        chunks = _per_var_chunks(plan)

        # Can't prune on non-dimension predicates, so all data chunks are expected.
        # The data variable has 105 chunks (7x5x3).
        assert chunks["data"] >= comprehensive_3d_dataset.total_chunks

    def test_filter_with_mixed_dim_and_non_dim_predicate(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """A filter combining dimension and non-dimension predicates should
        still prune based on the dimension predicate."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        pred = (pl.col("a") == 15) & (pl.col("data") > 0.5)
        cols = ["a", "b", "c", "data"]
        filter_expr = pl.col(cols).filter(pred)

        plan = backend.selected_chunks_debug(filter_expr)
        chunks = _per_var_chunks(plan)

        # Should prune on 'a' dimension: 1 chunk on a, 5 on b, 3 on c = 15
        assert chunks["data"] <= 15, (
            f"Expected at most 15 chunks with a==15, got {chunks['data']}. "
            f"Dimension predicate is not being applied."
        )

    def test_empty_filter_produces_no_chunks(
        self,
        comprehensive_3d_dataset: ComprehensiveDatasetInfo,
    ) -> None:
        """A filter with a provably-empty predicate should produce no chunks."""
        url = comprehensive_3d_dataset.path
        backend = ZarrBackend.from_url(url)

        # Contradictory predicate: a < 0 AND a > 100 → empty
        pred = (pl.col("a") < 0) & (pl.col("a") > 100)
        cols = ["a", "b", "c", "data"]
        filter_expr = pl.col(cols).filter(pred)

        plan = backend.selected_chunks_debug(filter_expr)
        total = _total_chunks(plan)

        assert total == 0, (
            f"Contradictory predicate should produce 0 chunks, got {total}"
        )
