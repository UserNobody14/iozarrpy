"""
Advanced Polars Expression Test Suite

Tests for:
- Column selectors and variable inference
- Aggregation expressions
- Window functions
- Array/struct operations
- Complex ternary with coordinate references
- Anonymous/mapped functions
- Selector set operations

These tests are designed for an IDEAL implementation where:
- Variables are inferred from expressions automatically
- Per-variable chunk selection is tracked
- The planner correctly handles complex expressions

All tests assert EXACT expected values. Tests for unimplemented features
are marked with xfail and will pass once the feature is implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import rainbear._core as _core
from rainbear import ZarrBackend

if TYPE_CHECKING:
    from conftest import MultiVarDatasetInfo

# Mark for tests that require unsupported expression handling
needs_expr_support = pytest.mark.xfail(
    reason="Expression type not yet supported in chunk planner",
    strict=False,
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_chunk_indices(zarr_url: str, expr: pl.Expr) -> set[tuple[int, ...]]:
    """Get the set of chunk indices selected by an expression."""
    grid_plans = ZarrBackend.from_url(zarr_url).selected_chunks_debug(expr)
    # Find a grid that includes "temp"
    for grid in grid_plans["grids"]:
        if "temp" in grid["variables"]:
            return {tuple(c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable 'temperature' in {grid_plans}")


def get_chunk_count(zarr_url: str, expr: pl.Expr) -> int:
    """Get the count of chunks selected by an expression."""
    return len(get_chunk_indices(zarr_url, expr))

def get_per_var_chunks(zarr_url: str, expr: pl.Expr) -> dict[str, int]:
    """Get per-variable chunk counts from an expression."""
    _, per_var, _ = _core._selected_variables_debug(zarr_url, expr)
    return {var: len(chunks) for var, chunks in per_var.items()}


# =============================================================================
# Test Classes
# =============================================================================


class TestColumnSelectors:
    """Tests for column selector expressions and basic variable inference."""

    def test_col_with_comparison_narrows_chunks(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """pl.col("a") < 10 should narrow to first chunk along 'a'."""
        expr = pl.col("a") < 10
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Should select 1 chunk in 'a' dimension, all in 'b' and 'c'
        # Expected: 1 * 4 * 3 = 12 chunks
        assert count == 12

    def test_col_with_range_comparison(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.col("a").is_between(10, 29) should select chunks 1 and 2."""
        expr = pl.col("a").is_between(10, 29)
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Should select 2 chunks in 'a' dimension (indices 10-19 and 20-29)
        # Expected: 2 * 4 * 3 = 24 chunks
        assert count == 24

    def test_multi_dim_constraint(self, multi_var_dataset: MultiVarDatasetInfo):
        """Constraint on two dimensions should narrow to intersection."""
        expr = (pl.col("a") < 10) & (pl.col("b") < 10)
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Should select 1 * 1 * 3 = 3 chunks
        assert count == 3

    def test_three_dim_constraint(self, multi_var_dataset: MultiVarDatasetInfo):
        """Constraint on all three dimensions should narrow to minimal set."""
        expr = (pl.col("a") < 10) & (pl.col("b") < 10) & (pl.col("c") < 10)
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Should select 1 * 1 * 1 = 1 chunk
        assert count == 1

    def test_is_in_multiple_values(self, multi_var_dataset: MultiVarDatasetInfo):
        """is_in with values in different chunks should select those chunks."""
        # Values 5, 15, 25 are in chunks 0, 1, 2 respectively (chunk_size=10)
        expr = pl.col("a").is_in([5, 15, 25])
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Should select 3 * 4 * 3 = 36 chunks
        assert count == 36


@pytest.mark.usefixtures("multi_var_dataset")
class TestAggregations:
    """Tests for aggregation expressions."""

    def test_sum_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum() aggregation needs all chunks."""
        expr = pl.col("temp").sum()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_mean_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """mean() aggregation needs all chunks."""
        expr = pl.col("precip").mean()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_min_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """min() aggregation needs all chunks."""
        expr = pl.col("wind_u").min()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_max_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """max() aggregation needs all chunks."""
        expr = pl.col("wind_v").max()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_count_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """count() aggregation needs all chunks."""
        expr = pl.col("pressure").count()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_std_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """std() aggregation needs all chunks."""
        expr = pl.col("temp").std()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_var_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """var() aggregation needs all chunks."""
        expr = pl.col("temp").var()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_filtered_sum_narrows_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum() with filter can narrow chunks."""
        expr = pl.col("temp").filter(pl.col("a") < 10).sum()
        count = get_chunk_count(multi_var_dataset.path, expr)
        # 1 * 4 * 3 = 12 chunks
        assert count == 12

    @needs_expr_support
    def test_filtered_mean_with_multi_dim(self, multi_var_dataset: MultiVarDatasetInfo):
        """mean() with multi-dim filter should narrow chunks."""
        expr = pl.col("temp").filter((pl.col("a") < 10) & (pl.col("b") < 10)).mean()
        count = get_chunk_count(multi_var_dataset.path, expr)
        # 1 * 1 * 3 = 3 chunks
        assert count == 3

    def test_n_unique_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """n_unique() needs all chunks."""
        expr = pl.col("temp").n_unique()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60


@pytest.mark.usefixtures("multi_var_dataset")
class TestWindowFunctions:
    """Tests for window function expressions."""

    def test_sum_over_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum().over() needs all chunks."""
        expr = pl.col("temp").sum().over("a")
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_mean_over_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """mean().over() needs all chunks."""
        expr = pl.col("precip").mean().over("b")
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_rank_over_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """rank().over() needs all chunks."""
        expr = pl.col("temp").rank().over("a")
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_rolling_mean(self, multi_var_dataset: MultiVarDatasetInfo):
        """rolling_mean() needs all chunks."""
        expr = pl.col("temp").rolling_mean(window_size=3)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_rolling_sum(self, multi_var_dataset: MultiVarDatasetInfo):
        """rolling_sum() needs all chunks."""
        expr = pl.col("temp").rolling_sum(window_size=5)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_cumsum(self, multi_var_dataset: MultiVarDatasetInfo):
        """cum_sum() needs all chunks."""
        expr = pl.col("temp").cum_sum()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_cummax(self, multi_var_dataset: MultiVarDatasetInfo):
        """cum_max() needs all chunks."""
        expr = pl.col("temp").cum_max()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_cummin(self, multi_var_dataset: MultiVarDatasetInfo):
        """cum_min() needs all chunks."""
        expr = pl.col("temp").cum_min()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_shift(self, multi_var_dataset: MultiVarDatasetInfo):
        """shift() needs all chunks (for correctness at boundaries)."""
        expr = pl.col("temp").shift(1)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    def test_diff(self, multi_var_dataset: MultiVarDatasetInfo):
        """diff() needs all chunks."""
        expr = pl.col("temp").diff()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

@pytest.mark.usefixtures("multi_var_dataset")
class TestComplexTernary:
    """Tests for complex ternary expressions with coordinate references."""
    @needs_expr_support
    def test_when_then_otherwise_with_null(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when().then().otherwise(null) should narrow to predicate chunks."""
        expr = pl.when(pl.col("a") < 10).then(pl.col("temp")).otherwise(pl.lit(None))
        count = get_chunk_count(multi_var_dataset.path, expr)
        # When otherwise is null, only need chunks where predicate is true
        # 1 * 4 * 3 = 12 chunks
        assert count == 12

    def test_when_then_otherwise_both_cols_all_chunks(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when().then().otherwise() with both branches needs all chunks."""
        expr = (
            pl.when(pl.col("a") < 20).then(pl.col("temp")).otherwise(pl.col("precip"))
        )
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Need all chunks because either branch could be taken
        assert count == 60

    @needs_expr_support
    def test_when_multi_condition_with_null(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when() with multi-dimension condition and null otherwise."""
        expr = (
            pl.when((pl.col("a") < 10) & (pl.col("b") < 10))
            .then(pl.col("temp"))
            .otherwise(pl.lit(None))
        )
        count = get_chunk_count(multi_var_dataset.path, expr)
        # 1 * 1 * 3 = 3 chunks
        assert count == 3


    def test_when_with_boolean_result(self, multi_var_dataset: MultiVarDatasetInfo):
        """when() returning boolean literals is equivalent to the predicate."""
        expr = pl.when(pl.col("a") < 10).then(pl.lit(True)).otherwise(pl.lit(False))
        count = get_chunk_count(multi_var_dataset.path, expr)
        # This is equivalent to pl.col("a") < 10
        # 1 * 4 * 3 = 12 chunks
        assert count == 12



