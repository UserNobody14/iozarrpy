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
import polars.selectors as cs
import pytest

from rainbear._core import _selected_chunks_debug, _selected_variables_debug

if TYPE_CHECKING:
    from conftest import MultiVarDatasetInfo


# Mark for tests that require variable inference (not yet implemented)
needs_var_inference = pytest.mark.xfail(
    reason="Variable inference from expressions not yet implemented",
    strict=False,
)

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
    chunks, _ = _selected_chunks_debug(zarr_url, expr, variables=["temp"])
    return {tuple(c["indices"]) for c in chunks}


def get_chunk_count(zarr_url: str, expr: pl.Expr) -> int:
    """Get the count of chunks selected by an expression."""
    return len(get_chunk_indices(zarr_url, expr))


def get_inferred_vars(zarr_url: str, expr: pl.Expr) -> set[str]:
    """Get the set of variables inferred from an expression."""
    inferred, _, _ = _selected_variables_debug(zarr_url, expr)
    return set(inferred)


def get_per_var_chunks(zarr_url: str, expr: pl.Expr) -> dict[str, int]:
    """Get per-variable chunk counts from an expression."""
    _, per_var, _ = _selected_variables_debug(zarr_url, expr)
    return {var: len(chunks) for var, chunks in per_var.items()}


# =============================================================================
# Test Classes
# =============================================================================


class TestColumnSelectors:
    """Tests for column selector expressions and basic variable inference."""

    @needs_var_inference
    def test_single_col_infers_variable(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.col("temp") should infer only the temp variable."""
        expr = pl.col("temp")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_var_inference
    def test_binary_op_infers_both_variables(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.col("temp") + pl.col("precip") should infer both variables."""
        expr = pl.col("temp") + pl.col("precip")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    @needs_var_inference
    def test_filter_with_coord_infers_data_and_coord(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """pl.col("temp").filter(pl.col("a") > 10) should infer temp and coord a."""
        expr = pl.col("temp").filter(pl.col("a") > 10)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a"}

    @needs_var_inference
    def test_multiple_cols_inferred(self, multi_var_dataset: MultiVarDatasetInfo):
        """Expression with multiple columns should infer all."""
        expr = pl.col("temp") * pl.col("wind_u") + pl.col("wind_v")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "wind_u", "wind_v"}

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

    @needs_var_inference
    def test_col_regex_pattern_match(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.col("^wind.*$") should match wind_u and wind_v."""
        expr = pl.col("^wind.*$")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u", "wind_v"}

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

    def test_cs_by_name(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.by_name("temp") should select only temp variable."""
        expr = cs.by_name("temp")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestVariableInference:
    """Tests for automatic variable detection from expressions."""

    @needs_var_inference
    def test_infer_single_var_from_col(self, multi_var_dataset: MultiVarDatasetInfo):
        """Single column reference should infer that variable."""
        expr = pl.col("precip")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"precip"}

    @needs_var_inference
    def test_infer_vars_from_arithmetic(self, multi_var_dataset: MultiVarDatasetInfo):
        """Arithmetic on columns should infer all referenced variables."""
        expr = pl.col("temp") - pl.col("pressure")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "pressure"}

    @needs_var_inference
    def test_infer_vars_from_comparison(self, multi_var_dataset: MultiVarDatasetInfo):
        """Comparison should infer the referenced variable."""
        expr = pl.col("wind_u") > 20
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u"}

    @needs_var_inference
    def test_infer_vars_from_chained_ops(self, multi_var_dataset: MultiVarDatasetInfo):
        """Chained operations should infer all variables."""
        expr = (pl.col("temp") + pl.col("precip")).abs() * pl.col("pressure")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "pressure"}

    @needs_var_inference
    def test_infer_coords_from_filter(self, multi_var_dataset: MultiVarDatasetInfo):
        """Filter on coord should infer the coord variable."""
        expr = pl.col("temp").filter(pl.col("a") > 30)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a"}

    @needs_var_inference
    def test_infer_2d_coord_from_filter(self, multi_var_dataset: MultiVarDatasetInfo):
        """Filter on 2D coord should infer that coord."""
        expr = pl.col("temp").filter(pl.col("lat") > 33.0)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "lat"}

    @needs_var_inference
    def test_per_var_chunk_counts_for_comparison(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """Per-variable chunk counts should be tracked separately."""
        expr = pl.col("a") < 10
        per_var = get_per_var_chunks(multi_var_dataset.path, expr)
        # 3D vars: 1 * 4 * 3 = 12 chunks each
        # 2D var (surface): 4 * 3 = 12 chunks (no 'a' dim, so no narrowing)
        assert per_var["temp"] == 12
        assert per_var["precip"] == 12
        assert per_var["wind_u"] == 12
        assert per_var["wind_v"] == 12
        assert per_var["pressure"] == 12
        assert per_var["surface"] == 12

    def test_comparison_chunks_for_3d_var(self, multi_var_dataset: MultiVarDatasetInfo):
        """Comparison on 'a' should narrow chunks for 3D variables."""
        expr = pl.col("a") < 20
        # Test chunk count using _selected_chunks_debug with explicit variable
        for var in ["temp", "precip", "wind_u", "wind_v", "pressure"]:
            chunks, _ = _selected_chunks_debug(multi_var_dataset.path, expr, variables=[var])
            count = len(chunks)
            # 3D vars should have 2 * 4 * 3 = 24 chunks
            assert count == 24, f"{var} should have 24 chunks, got {count}"

    def test_literal_expr_includes_all_vars(self, multi_var_dataset: MultiVarDatasetInfo):
        """Literal true should include all variables with full chunk counts."""
        expr = pl.lit(True)
        per_var = get_per_var_chunks(multi_var_dataset.path, expr)
        # 3D variables should have 60 chunks each
        for var in multi_var_dataset.vars_3d:
            if var in per_var:
                assert per_var[var] == 60, f"{var} should have 60 chunks"
        # 2D variable should have 12 chunks
        if "surface" in per_var:
            assert per_var["surface"] == 12

    @needs_var_inference
    def test_alias_preserves_inference(self, multi_var_dataset: MultiVarDatasetInfo):
        """alias() should not affect variable inference."""
        expr = pl.col("temp").alias("temperature")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_var_inference
    def test_cast_preserves_inference(self, multi_var_dataset: MultiVarDatasetInfo):
        """cast() should not affect variable inference."""
        expr = pl.col("temp").cast(pl.Float32)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_var_inference
    def test_abs_preserves_inference(self, multi_var_dataset: MultiVarDatasetInfo):
        """abs() should preserve variable inference."""
        expr = pl.col("wind_v").abs()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_v"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestAggregations:
    """Tests for aggregation expressions."""

    @needs_expr_support
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

    @needs_expr_support
    def test_std_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """std() aggregation needs all chunks."""
        expr = pl.col("temp").std()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
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

    @needs_expr_support
    def test_sum_infers_variable(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum() should infer the aggregated variable."""
        expr = pl.col("wind_u").sum()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u"}

    @needs_expr_support
    def test_n_unique_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """n_unique() needs all chunks."""
        expr = pl.col("temp").n_unique()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60


@pytest.mark.usefixtures("multi_var_dataset")
class TestWindowFunctions:
    """Tests for window function expressions."""

    @needs_expr_support
    def test_sum_over_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum().over() needs all chunks."""
        expr = pl.col("temp").sum().over("a")
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_sum_over_infers_both_vars(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum().over() should infer data and partition columns."""
        expr = pl.col("temp").sum().over("a")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a"}

    @needs_expr_support
    def test_mean_over_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """mean().over() needs all chunks."""
        expr = pl.col("precip").mean().over("b")
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_rank_over_needs_all_chunks(self, multi_var_dataset: MultiVarDatasetInfo):
        """rank().over() needs all chunks."""
        expr = pl.col("temp").rank().over("a")
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_rolling_mean(self, multi_var_dataset: MultiVarDatasetInfo):
        """rolling_mean() needs all chunks."""
        expr = pl.col("temp").rolling_mean(window_size=3)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_rolling_sum(self, multi_var_dataset: MultiVarDatasetInfo):
        """rolling_sum() needs all chunks."""
        expr = pl.col("temp").rolling_sum(window_size=5)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_cumsum(self, multi_var_dataset: MultiVarDatasetInfo):
        """cum_sum() needs all chunks."""
        expr = pl.col("temp").cum_sum()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_cummax(self, multi_var_dataset: MultiVarDatasetInfo):
        """cum_max() needs all chunks."""
        expr = pl.col("temp").cum_max()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_cummin(self, multi_var_dataset: MultiVarDatasetInfo):
        """cum_min() needs all chunks."""
        expr = pl.col("temp").cum_min()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_shift(self, multi_var_dataset: MultiVarDatasetInfo):
        """shift() needs all chunks (for correctness at boundaries)."""
        expr = pl.col("temp").shift(1)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_diff(self, multi_var_dataset: MultiVarDatasetInfo):
        """diff() needs all chunks."""
        expr = pl.col("temp").diff()
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_over_multiple_partition_cols(self, multi_var_dataset: MultiVarDatasetInfo):
        """over() with multiple partition cols should infer all."""
        expr = pl.col("temp").rank().over(["a", "b"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a", "b"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestArrayStructOps:
    """Tests for array and struct operations."""

    @needs_expr_support
    def test_struct_creation(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.struct() should infer all member variables."""
        expr = pl.struct(["temp", "precip"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    @needs_expr_support
    def test_struct_three_members(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.struct() with three members should infer all."""
        expr = pl.struct(["temp", "precip", "wind_u"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "wind_u"}

    @needs_expr_support
    def test_concat_list(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.concat_list() should infer all member variables."""
        expr = pl.concat_list(["temp", "precip"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    @needs_expr_support
    def test_implode(self, multi_var_dataset: MultiVarDatasetInfo):
        """implode() should infer the column variable."""
        expr = pl.col("temp").implode()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_expr_support
    def test_struct_field_access(self, multi_var_dataset: MultiVarDatasetInfo):
        """struct.field() should conservatively infer all struct members."""
        expr = pl.struct(["a", "b"]).struct.field("a")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # Conservative: both a and b should be inferred since we built the struct
        assert inferred == {"a", "b"}

    @needs_expr_support
    def test_list_len(self, multi_var_dataset: MultiVarDatasetInfo):
        """list.len() should infer the list column."""
        expr = pl.col("temp").implode().list.len()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_expr_support
    def test_list_first(self, multi_var_dataset: MultiVarDatasetInfo):
        """list.first() should infer the list column."""
        expr = pl.col("temp").implode().list.first()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_expr_support
    def test_list_last(self, multi_var_dataset: MultiVarDatasetInfo):
        """list.last() should infer the list column."""
        expr = pl.col("temp").implode().list.last()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestComplexTernary:
    """Tests for complex ternary expressions with coordinate references."""

    @needs_expr_support
    def test_when_then_otherwise_with_null(self, multi_var_dataset: MultiVarDatasetInfo):
        """when().then().otherwise(null) should narrow to predicate chunks."""
        expr = pl.when(pl.col("a") < 10).then(pl.col("temp")).otherwise(pl.lit(None))
        count = get_chunk_count(multi_var_dataset.path, expr)
        # When otherwise is null, only need chunks where predicate is true
        # 1 * 4 * 3 = 12 chunks
        assert count == 12

    @needs_expr_support
    def test_when_then_otherwise_both_cols(self, multi_var_dataset: MultiVarDatasetInfo):
        """when().then().otherwise() with both branches referencing columns."""
        expr = pl.when(pl.col("a") < 20).then(pl.col("temp")).otherwise(pl.col("precip"))
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "a"}

    @needs_expr_support
    def test_when_then_otherwise_both_cols_all_chunks(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when().then().otherwise() with both branches needs all chunks."""
        expr = pl.when(pl.col("a") < 20).then(pl.col("temp")).otherwise(pl.col("precip"))
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Need all chunks because either branch could be taken
        assert count == 60

    @needs_expr_support
    def test_when_multi_condition_with_null(self, multi_var_dataset: MultiVarDatasetInfo):
        """when() with multi-dimension condition and null otherwise."""
        expr = (
            pl.when((pl.col("a") < 10) & (pl.col("b") < 10))
            .then(pl.col("temp"))
            .otherwise(pl.lit(None))
        )
        count = get_chunk_count(multi_var_dataset.path, expr)
        # 1 * 1 * 3 = 3 chunks
        assert count == 3

    @needs_expr_support
    def test_when_coord_reference_in_predicate(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when() with 2D coord in predicate."""
        expr = (
            pl.when(pl.col("lat") > 33.0)
            .then(pl.col("temp"))
            .otherwise(pl.lit(0))
        )
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "lat"}

    @needs_expr_support
    def test_chained_when_then(self, multi_var_dataset: MultiVarDatasetInfo):
        """Chained when().then().when().then().otherwise()."""
        expr = (
            pl.when(pl.col("a") < 10)
            .then(pl.col("temp"))
            .when(pl.col("a") < 20)
            .then(pl.col("precip"))
            .otherwise(pl.col("wind_u"))
        )
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "wind_u", "a"}

    @needs_expr_support
    def test_nested_when(self, multi_var_dataset: MultiVarDatasetInfo):
        """Nested when() expressions."""
        inner = pl.when(pl.col("b") < 10).then(pl.col("precip")).otherwise(pl.lit(0))
        expr = pl.when(pl.col("a") < 10).then(inner).otherwise(pl.lit(0))
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"precip", "a", "b"}

    def test_when_with_boolean_result(self, multi_var_dataset: MultiVarDatasetInfo):
        """when() returning boolean literals is equivalent to the predicate."""
        expr = (
            pl.when(pl.col("a") < 10)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
        )
        count = get_chunk_count(multi_var_dataset.path, expr)
        # This is equivalent to pl.col("a") < 10
        # 1 * 4 * 3 = 12 chunks
        assert count == 12

    @needs_expr_support
    def test_when_lon_lat_predicate(self, multi_var_dataset: MultiVarDatasetInfo):
        """when() with lon/lat predicate selecting specific region."""
        # lon is -120 + c*0.1, lat is 30 + b*0.1
        # lon > -118 means c > 20 (chunk 2)
        # lat > 32 means b > 20 (chunks 2, 3)
        expr = (
            pl.when((pl.col("lon") > -118.0) & (pl.col("lat") > 32.0))
            .then(pl.col("temp"))
            .otherwise(pl.lit(None))
        )
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "lon", "lat"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestAnonymousFunctions:
    """Tests for anonymous/mapped functions (conservative behavior)."""

    @needs_expr_support
    def test_map_elements_conservative(self, multi_var_dataset: MultiVarDatasetInfo):
        """map_elements() should conservatively return all chunks."""
        expr = pl.col("temp").map_elements(lambda x: x * 2, return_dtype=pl.Float64)
        count = get_chunk_count(multi_var_dataset.path, expr)
        # Cannot push down, must return all
        assert count == 60

    @needs_expr_support
    def test_map_elements_infers_variable(self, multi_var_dataset: MultiVarDatasetInfo):
        """map_elements() should still infer the variable."""
        expr = pl.col("precip").map_elements(lambda x: x + 1, return_dtype=pl.Float64)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"precip"}

    @needs_expr_support
    def test_map_batches_conservative(self, multi_var_dataset: MultiVarDatasetInfo):
        """map_batches() should conservatively return all chunks."""
        expr = pl.col("temp").map_batches(lambda s: s * 2)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60

    @needs_expr_support
    def test_map_batches_infers_variable(self, multi_var_dataset: MultiVarDatasetInfo):
        """map_batches() should still infer the variable."""
        expr = pl.col("temp").map_batches(lambda s: s.abs())
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    @needs_expr_support
    def test_nested_map_elements(self, multi_var_dataset: MultiVarDatasetInfo):
        """Nested map_elements should be conservative."""
        expr = pl.col("temp").map_elements(
            lambda x: x * 2, return_dtype=pl.Float64
        ).map_elements(lambda x: x + 1, return_dtype=pl.Float64)
        count = get_chunk_count(multi_var_dataset.path, expr)
        assert count == 60


@pytest.mark.usefixtures("multi_var_dataset")
class TestSelectorSetOps:
    """Tests for selector set operations."""

    def test_selector_union(self, multi_var_dataset: MultiVarDatasetInfo):
        """Union of selectors should combine variables."""
        expr = cs.by_name("temp") | cs.by_name("precip")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    @needs_var_inference
    def test_selector_difference(self, multi_var_dataset: MultiVarDatasetInfo):
        """Difference of selectors should exclude variables."""
        expr = cs.all() - cs.by_name("surface")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # All data vars except surface
        expected = {"temp", "precip", "wind_u", "wind_v", "pressure"}
        assert inferred == expected

    def test_selector_intersection(self, multi_var_dataset: MultiVarDatasetInfo):
        """Intersection of selectors should find common variables."""
        expr = cs.numeric() & cs.by_name("temp", "precip", "surface")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "surface"}

    def test_selector_exclusive_or(self, multi_var_dataset: MultiVarDatasetInfo):
        """XOR of selectors."""
        expr = cs.by_name("temp", "precip") ^ cs.by_name("precip", "wind_u")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # temp and wind_u (not precip - it's in both)
        assert inferred == {"temp", "wind_u"}

    @needs_var_inference
    def test_cs_numeric(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.numeric() should select all numeric columns."""
        expr = cs.numeric()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # All data vars are numeric
        expected = {"temp", "precip", "wind_u", "wind_v", "pressure", "surface"}
        assert inferred == expected

    @needs_var_inference
    def test_cs_float(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.float() should select float columns."""
        expr = cs.float()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # All data vars are float64
        expected = {"temp", "precip", "wind_u", "wind_v", "pressure", "surface"}
        assert inferred == expected

    @needs_var_inference
    def test_cs_matches_regex(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.matches() with regex pattern."""
        expr = cs.matches("^wind")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u", "wind_v"}

    @needs_var_inference
    def test_complex_selector_combo(self, multi_var_dataset: MultiVarDatasetInfo):
        """Complex combination of selectors."""
        # numeric & matches(temp|precip) - surface
        # = {temp, precip} (surface doesn't match the regex anyway)
        expr = (cs.numeric() & cs.matches("^temp|^precip")) - cs.by_name("surface")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}
