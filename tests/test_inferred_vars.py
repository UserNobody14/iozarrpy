from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
import pytest

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from conftest import MultiVarDatasetInfo

    from rainbear._core import SelectedChunksDebugReturn



# =============================================================================
# Helper Functions
# =============================================================================


def get_inferred_vars(zarr_url: str, expr: pl.Expr) -> set[str]:
    """Get the set of variables inferred from an expression."""
    inferred = ZarrBackend.from_url(zarr_url).selected_chunks_debug(expr)
    list_of_vars: list[list[str]] = [grid["variables"] for grid in inferred["grids"]]
    # Combine all the variables from the grids
    return set(chain.from_iterable(list_of_vars))

def get_inferred_dims(zarr_url: str, expr: pl.Expr) -> set[str]:
    """Get the set of dimensions inferred from an expression."""
    inferred = ZarrBackend.from_url(zarr_url).selected_chunks_debug(expr)
    list_of_dims: list[list[str]] = [grid["dims"] for grid in inferred["grids"]]
    # Combine all the dimensions from the grids
    return set(chain.from_iterable(list_of_dims))


def get_per_var_chunks(zarr_url: str, expr: pl.Expr) -> dict[str, int]:
    """Get per-variable chunk counts from an expression."""
    grid_plans = ZarrBackend.from_url(zarr_url).selected_chunks_debug(expr)
    # For each grid, get the variables and count the chunks
    per_var = {}
    for grid in grid_plans["grids"]:
        for var in grid["variables"]:
            per_var[var] = len(grid["chunks"])
    return per_var


def _chunk_indices(chunks: SelectedChunksDebugReturn, variable: str) -> set[tuple[int, ...]]:
    """Get the chunk indices for a variable."""
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")

@pytest.mark.usefixtures("multi_var_dataset")
class TestArrayStructOps:
    """Tests for array and struct operations."""

    def test_struct_creation(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.struct() should infer all member variables."""
        expr = pl.struct(["temp", "precip"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    def test_struct_three_members(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.struct() with three members should infer all."""
        expr = pl.struct(["temp", "precip", "wind_u"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "wind_u"}

    def test_concat_list(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.concat_list() should infer all member variables."""
        expr = pl.concat_list(["temp", "precip"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    def test_implode(self, multi_var_dataset: MultiVarDatasetInfo):
        """implode() should infer the column variable."""
        expr = pl.col("temp").implode()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    def test_struct_field_access(self, multi_var_dataset: MultiVarDatasetInfo):
        """struct.field() should conservatively infer all struct members."""
        expr = pl.struct(["a", "b"]).struct.field("a")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # Conservative: both a and b should be inferred since we built the struct
        assert inferred == {"a", "b"}

    def test_list_len(self, multi_var_dataset: MultiVarDatasetInfo):
        """list.len() should infer the list column."""
        expr = pl.col("temp").implode().list.len()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    def test_list_first(self, multi_var_dataset: MultiVarDatasetInfo):
        """list.first() should infer the list column."""
        expr = pl.col("temp").implode().list.first()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    def test_list_last(self, multi_var_dataset: MultiVarDatasetInfo):
        """list.last() should infer the list column."""
        expr = pl.col("temp").implode().list.last()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestSelectorSetOps:
    """Tests for selector set operations."""

    def test_selector_union(self, multi_var_dataset: MultiVarDatasetInfo):
        """Union of selectors should combine variables."""
        expr = cs.by_name("temp") | cs.by_name("precip")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

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

    def test_cs_numeric(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.numeric() should select all numeric columns."""
        expr = cs.numeric()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # All data vars are numeric
        expected = {"temp", "precip", "wind_u", "wind_v", "pressure", "surface"}
        assert inferred == expected

    def test_cs_float(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.float() should select float columns."""
        expr = cs.float()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        # All data vars are float64
        expected = {"temp", "precip", "wind_u", "wind_v", "pressure", "surface"}
        assert inferred == expected

    def test_cs_matches_regex(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.matches() with regex pattern."""
        expr = cs.matches("^wind")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u", "wind_v"}

    def test_complex_selector_combo(self, multi_var_dataset: MultiVarDatasetInfo):
        """Complex combination of selectors."""
        # numeric & matches(temp|precip) - surface
        # = {temp, precip} (surface doesn't match the regex anyway)
        expr = (cs.numeric() & cs.matches("^temp|^precip")) - cs.by_name("surface")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestTernary:

    def test_when_then_otherwise_both_cols(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when().then().otherwise() with both branches referencing columns."""
        expr = (
            pl.when(pl.col("a") < 20).then(pl.col("temp")).otherwise(pl.col("precip"))
        )
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "a"}


    def test_when_coord_reference_in_predicate(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """when() with 2D coord in predicate."""
        expr = pl.when(pl.col("lat") > 33.0).then(pl.col("temp")).otherwise(pl.lit(0))
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "lat"}

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

    def test_nested_when(self, multi_var_dataset: MultiVarDatasetInfo):
        """Nested when() expressions."""
        inner = pl.when(pl.col("b") < 10).then(pl.col("precip")).otherwise(pl.lit(0))
        expr = pl.when(pl.col("a") < 10).then(inner).otherwise(pl.lit(0))
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"precip"}

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
        assert inferred == {"temp"}


@pytest.mark.usefixtures("multi_var_dataset")
class TestVariableInference:
    """Tests for automatic variable detection from expressions."""

    def test_infer_single_var_from_col(self, multi_var_dataset: MultiVarDatasetInfo):
        """Single column reference should infer that variable."""
        expr = pl.col("precip")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"precip"}

    def test_infer_vars_from_arithmetic(self, multi_var_dataset: MultiVarDatasetInfo):
        """Arithmetic on columns should infer all referenced variables."""
        expr = pl.col("temp") - pl.col("pressure")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "pressure"}

    def test_infer_vars_from_chained_ops(self, multi_var_dataset: MultiVarDatasetInfo):
        """Chained operations should infer all variables."""
        expr = (pl.col("temp") + pl.col("precip")).abs() * pl.col("pressure")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip", "pressure"}

    def test_infer_coords_from_filter(self, multi_var_dataset: MultiVarDatasetInfo):
        """Filter on coord should infer the coord variable."""
        expr = pl.col("temp").filter(pl.col("a") > 30)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        inferred_dims = get_inferred_dims(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a"}
        assert inferred_dims == {"a", "b", "c"}

    def test_infer_2d_coord_from_filter(self, multi_var_dataset: MultiVarDatasetInfo):
        """Filter on 2D coord should infer that coord."""
        expr = pl.col("temp").filter((pl.col("a") > 33.0) & (pl.col("b") > 33.0))
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        inferred_dims = get_inferred_dims(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a", "b"}
        assert inferred_dims == {"a", "b", "c"}

    def test_alias_preserves_inference(self, multi_var_dataset: MultiVarDatasetInfo):
        """alias() should not affect variable inference."""
        expr = pl.col("temp").alias("temperature")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    def test_cast_preserves_inference(self, multi_var_dataset: MultiVarDatasetInfo):
        """cast() should not affect variable inference."""
        expr = pl.col("temp").cast(pl.Float32)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    def test_abs_preserves_inference(self, multi_var_dataset: MultiVarDatasetInfo):
        """abs() should preserve variable inference."""
        expr = pl.col("wind_v").abs()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_v"}

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
            chunks = ZarrBackend.from_url(multi_var_dataset.path).selected_chunks_debug(expr)
            idxs = _chunk_indices(chunks, variable=var)
            count = len(idxs)
            # 3D vars should have 2 * 4 * 3 = 24 chunks
            assert count == 24, f"{var} should have 24 chunks, got {count} in {chunks}"

    def test_literal_expr_includes_all_vars(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
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



@pytest.mark.usefixtures("multi_var_dataset")
class TestWindowFunctions:
    """Tests for window function expressions."""


    def test_sum_over_infers_both_vars(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum().over() should infer data and partition columns."""
        expr = pl.col("temp").sum().over("a")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a"}


    def test_over_multiple_partition_cols(self, multi_var_dataset: MultiVarDatasetInfo):
        """over() with multiple partition cols should infer all."""
        expr = pl.col("temp").rank().over(["a", "b"])
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a", "b"}



@pytest.mark.usefixtures("multi_var_dataset")
class TestAggregations:
    """Tests for aggregation expressions."""


    def test_sum_infers_variable(self, multi_var_dataset: MultiVarDatasetInfo):
        """sum() should infer the aggregated variable."""
        expr = pl.col("wind_u").sum()
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u"}

class TestColumnSelectors:
    """Tests for column selector expressions and basic variable inference."""

    def test_single_col_infers_variable(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.col("temp") should infer only the temp variable."""
        expr = pl.col("temp")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}

    def test_binary_op_infers_both_variables(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """pl.col("temp") + pl.col("precip") should infer both variables."""
        expr = pl.col("temp") + pl.col("precip")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "precip"}

    def test_filter_with_coord_infers_data_and_coord(
        self, multi_var_dataset: MultiVarDatasetInfo
    ):
        """pl.col("temp").filter(pl.col("a") > 10) should infer temp and coord a."""
        expr = pl.col("temp").filter(pl.col("a") > 10)
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "a"}

    def test_multiple_cols_inferred(self, multi_var_dataset: MultiVarDatasetInfo):
        """Expression with multiple columns should infer all."""
        expr = pl.col("temp") * pl.col("wind_u") + pl.col("wind_v")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp", "wind_u", "wind_v"}


    def test_col_regex_pattern_match(self, multi_var_dataset: MultiVarDatasetInfo):
        """pl.col("^wind.*$") should match wind_u and wind_v."""
        expr = pl.col("^wind.*$")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"wind_u", "wind_v"}


    def test_cs_by_name(self, multi_var_dataset: MultiVarDatasetInfo):
        """cs.by_name("temp") should select only temp variable."""
        expr = cs.by_name("temp")
        inferred = get_inferred_vars(multi_var_dataset.path, expr)
        assert inferred == {"temp"}
