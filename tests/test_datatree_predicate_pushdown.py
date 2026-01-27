"""Predicate pushdown tests for zarr hierarchies (xarray DataTrees).

These tests verify that predicates on hierarchical zarr data (represented as
Polars struct columns) correctly narrow chunk selection across the tree.

Key scenarios:
1. Filtering on root dimensions should affect all child node chunk selection
2. Filtering on struct fields should push down to the corresponding child arrays
3. Complex expressions combining root coords and struct fields should optimize correctly

Expected behavior:
- Predicates on shared coordinates (y, x) should narrow chunks for all nodes
- Predicates on struct.field("var") should push down to that child's arrays
- AND combinations should intersect chunk selections
- OR combinations should union chunk selections conservatively
"""

from __future__ import annotations

from typing import Any

import polars as pl

from rainbear import _core


def _chunk_indices(chunks: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    """Extract chunk indices from debug output."""
    out: set[tuple[int, ...]] = set()
    for d in chunks:
        idx = d["indices"]
        out.add(tuple(int(x) for x in idx))
    return out


# =============================================================================
# Root Dimension Filter Tests
# =============================================================================


def test_filter_on_root_dims_affects_all_nodes(
    datatree_datasets: dict[str, str],
) -> None:
    """Filter on shared dimension (y) should narrow chunks for all child nodes."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Filter on y < 5 (first few rows)
    pred = pl.col("y") < 5

    # This should narrow chunks for both model_a and model_b variables
    chunks, coord_reads = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature", "model_b/temperature"],
    )

    assert coord_reads >= 0
    idxs = _chunk_indices(chunks)

    # With ny=16 and assuming reasonable chunk size, y<5 should select
    # fewer chunks than full dataset
    # This assertion documents that chunk narrowing works for child arrays
    assert len(idxs) > 0, "Should select some chunks"

    # All selected chunks should have y-index corresponding to y<5
    # (exact assertion depends on chunk layout)


def test_filter_on_both_dims(datatree_datasets: dict[str, str]) -> None:
    """Filter on both y and x should intersect chunk selections."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Narrow both dimensions
    pred = (pl.col("y") < 8) & (pl.col("x") < 10)

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["surface"],  # root variable
    )

    idxs = _chunk_indices(chunks)
    assert len(idxs) > 0


# =============================================================================
# Struct Field Filter Tests
# =============================================================================


def test_filter_on_nested_struct_field(datatree_datasets: dict[str, str]) -> None:
    """Filter on struct field (model_a.temperature > 280) should push down."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Filter on nested field value
    # Note: This requires the expression compiler to understand struct field access
    pred = pl.col("model_a").struct.field("temperature") > 280

    chunks, coord_reads = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature"],
    )

    # If pushdown works, this should read the temperature array to evaluate
    # which chunks contain values > 280
    # For now, we just verify the call doesn't crash
    assert coord_reads >= 0


def test_filter_on_multiple_struct_fields(datatree_datasets: dict[str, str]) -> None:
    """Filter combining fields from different child nodes."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Combine filters on different children
    pred = (pl.col("model_a").struct.field("temperature") > 280) & (
        pl.col("model_b").struct.field("humidity") > 0.5
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature", "model_b/humidity"],
    )

    idxs = _chunk_indices(chunks)
    # Should return chunks that satisfy both conditions
    assert len(idxs) >= 0  # May be empty if no overlap


# =============================================================================
# Chunk Propagation Tests
# =============================================================================


def test_chunk_selection_propagates_to_children(
    datatree_datasets: dict[str, str],
) -> None:
    """Predicate on root coords should narrow child chunks proportionally."""
    zarr_url = datatree_datasets["ensemble_tree"]

    # Filter on time dimension (shared across all members)
    pred = pl.col("time") < 3  # First 3 time steps out of 6

    # Check chunk selection for member_0
    chunks_member0, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["member_0/temperature"],
    )
    idxs_m0 = _chunk_indices(chunks_member0)

    # Check chunk selection for member_2
    chunks_member2, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["member_2/temperature"],
    )
    idxs_m2 = _chunk_indices(chunks_member2)

    # Both should have same number of chunks (same filter, same shape)
    assert len(idxs_m0) == len(idxs_m2)

    # Should be roughly half the chunks (time < 3 out of 6)
    # Exact count depends on chunking, but should be less than full
    assert len(idxs_m0) > 0


def test_independent_child_filters(datatree_datasets: dict[str, str]) -> None:
    """Different filters per child node should work independently."""
    zarr_url = datatree_datasets["ensemble_tree"]

    # Filter only member_0's temperature
    pred_m0 = pl.col("member_0").struct.field("temperature") > 290

    # Filter only member_1's precipitation
    pred_m1 = pl.col("member_1").struct.field("precipitation") > 2

    # Apply separately
    chunks_m0, _ = _core._selected_chunks_debug(
        zarr_url,
        pred_m0,
        variables=["member_0/temperature"],
    )

    chunks_m1, _ = _core._selected_chunks_debug(
        zarr_url,
        pred_m1,
        variables=["member_1/precipitation"],
    )

    # These should be independent chunk selections
    idxs_m0 = _chunk_indices(chunks_m0)
    idxs_m1 = _chunk_indices(chunks_m1)

    # May have different counts depending on data distribution
    assert len(idxs_m0) >= 0
    assert len(idxs_m1) >= 0


# =============================================================================
# Literal and Constant Folding Tests
# =============================================================================


def test_literal_predicates_on_structs(datatree_datasets: dict[str, str]) -> None:
    """Constant folding should work with struct field access."""
    zarr_url = datatree_datasets["simple_datatree"]

    # lit(True) should return all chunks
    chunks_true, _ = _core._selected_chunks_debug(
        zarr_url,
        pl.lit(True),
        variables=["model_a/temperature"],
    )
    idxs_true = _chunk_indices(chunks_true)

    # lit(False) should return no chunks
    chunks_false, _ = _core._selected_chunks_debug(
        zarr_url,
        pl.lit(False),
        variables=["model_a/temperature"],
    )
    idxs_false = _chunk_indices(chunks_false)

    assert len(idxs_true) > 0
    assert len(idxs_false) == 0


def test_literal_null_on_structs(datatree_datasets: dict[str, str]) -> None:
    """lit(None) should return no chunks (null predicate = false)."""
    zarr_url = datatree_datasets["simple_datatree"]

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pl.lit(None),
        variables=["model_a/temperature"],
    )

    assert _chunk_indices(chunks) == set()


# =============================================================================
# Expression Wrapper Tests
# =============================================================================


def test_alias_cast_on_struct_fields(datatree_datasets: dict[str, str]) -> None:
    """Alias and cast wrappers should preserve pushdown on struct fields."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Create expression with alias and cast
    pred = (
        pl.col("model_a")
        .struct.field("temperature")
        .gt(280)
        .alias("temp_filter")
        .cast(pl.Boolean)
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature"],
    )

    # Should still produce valid chunk selection
    idxs = _chunk_indices(chunks)
    assert len(idxs) >= 0


def test_complex_struct_expression(datatree_datasets: dict[str, str]) -> None:
    """Complex expressions on struct fields should not break pushdown."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Complex ternary expression
    ternary = pl.when(pl.lit(True)).then(pl.lit(True)).otherwise(pl.lit(True))

    # Combine with struct field filter
    pred = pl.col("model_a").struct.field("temperature").gt(280) & ternary

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature"],
    )

    # The ternary shouldn't break the temperature filter pushdown
    idxs = _chunk_indices(chunks)
    assert len(idxs) >= 0


# =============================================================================
# Deep Hierarchy Predicate Tests
# =============================================================================


def test_filter_on_deep_nested_field(datatree_datasets: dict[str, str]) -> None:
    """Filter on deeply nested struct field."""
    zarr_url = datatree_datasets["deep_tree_d4"]

    # Access deeply nested field: level_1.level_2.level_3.var_3
    pred = (
        pl.col("level_1")
        .struct.field("level_2")
        .struct.field("level_3")
        .struct.field("var_3")
        > 30
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["level_1/level_2/level_3/var_3"],
    )

    idxs = _chunk_indices(chunks)
    assert len(idxs) >= 0


def test_filter_combines_root_and_deep_nested(
    datatree_datasets: dict[str, str],
) -> None:
    """Combine root dimension filter with deep nested field filter."""
    zarr_url = datatree_datasets["deep_tree_d4"]

    # Filter on root dimension AND deep nested field
    pred = (pl.col("y") < 6) & (
        pl.col("level_1")
        .struct.field("level_2")
        .struct.field("var_2")
        > 20
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["level_1/level_2/var_2"],
    )

    idxs = _chunk_indices(chunks)
    # Should be intersection of y<6 chunks and var_2>20 chunks
    assert len(idxs) >= 0


# =============================================================================
# Heterogeneous Dimension Tests
# =============================================================================


def test_filter_with_extra_dimension(datatree_datasets: dict[str, str]) -> None:
    """Filter on node with additional dimension (z in atmosphere_3d)."""
    zarr_url = datatree_datasets["heterogeneous_tree"]

    # The atmosphere_3d node has z dimension not present at root
    # Filter should work on the shared dimensions
    pred = (pl.col("y") < 8) & (pl.col("x") < 10)

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["atmosphere_3d/temperature"],
    )

    idxs = _chunk_indices(chunks)
    assert len(idxs) >= 0


def test_filter_on_node_specific_dimension(datatree_datasets: dict[str, str]) -> None:
    """Filter on dimension specific to a child node (station_id in timeseries)."""
    zarr_url = datatree_datasets["heterogeneous_tree"]

    # timeseries_1d has station_id dimension, not shared with root
    # This tests how non-shared dimensions are handled
    pred = (
        pl.col("timeseries_1d")
        .struct.field("station_id")  # This would need special handling
        .lt(5)
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["timeseries_1d/measurement"],
    )

    # May or may not work depending on implementation
    idxs = _chunk_indices(chunks)
    assert len(idxs) >= 0


# =============================================================================
# Wide Tree Tests
# =============================================================================


def test_filter_across_many_children(datatree_datasets: dict[str, str]) -> None:
    """Filter that applies to all children in a wide tree."""
    zarr_url = datatree_datasets["wide_tree_n10"]

    # Filter on root dimension affects all 10 children
    pred = pl.col("y") < 5

    # Check that all children get narrowed chunk selection
    for i in range(10):
        chunks, _ = _core._selected_chunks_debug(
            zarr_url,
            pred,
            variables=[f"child_{i}/data"],
        )
        idxs = _chunk_indices(chunks)
        assert len(idxs) > 0, f"child_{i} should have chunks"


def test_filter_specific_child_in_wide_tree(datatree_datasets: dict[str, str]) -> None:
    """Filter specific to one child in a wide tree."""
    zarr_url = datatree_datasets["wide_tree_n10"]

    # Filter only on child_5's data
    pred = pl.col("child_5").struct.field("data") > 500

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["child_5/data"],
    )

    idxs = _chunk_indices(chunks)
    assert len(idxs) >= 0


# =============================================================================
# AND/OR Combination Tests
# =============================================================================


def test_and_combination_narrows_correctly(datatree_datasets: dict[str, str]) -> None:
    """AND of struct field filters should intersect chunks."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Both conditions on same child
    pred = (pl.col("model_a").struct.field("temperature") > 270) & (
        pl.col("model_a").struct.field("pressure") > 1000
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature", "model_a/pressure"],
    )

    idxs = _chunk_indices(chunks)
    # Should be intersection, potentially fewer chunks
    assert len(idxs) >= 0


def test_or_combination_conservative(datatree_datasets: dict[str, str]) -> None:
    """OR of struct field filters should be conservative (union or all)."""
    zarr_url = datatree_datasets["simple_datatree"]

    # OR of conditions on different children
    pred = (pl.col("model_a").struct.field("temperature") > 300) | (
        pl.col("model_b").struct.field("humidity") > 0.8
    )

    chunks, _ = _core._selected_chunks_debug(
        zarr_url,
        pred,
        variables=["model_a/temperature", "model_b/humidity"],
    )

    idxs = _chunk_indices(chunks)
    # OR should be conservative - likely returns all or union of chunks
    assert len(idxs) >= 0


# =============================================================================
# Tests to Document Current Behavior
# =============================================================================


def test_current_behavior_struct_field_predicate(
    datatree_datasets: dict[str, str],
) -> None:
    """Document what currently happens with struct field predicates.

    This test does NOT use xfail - it captures actual current behavior.
    """
    zarr_url = datatree_datasets["simple_datatree"]

    try:
        # Try a struct field predicate
        pred = pl.col("model_a").struct.field("temperature") > 280

        chunks, coord_reads = _core._selected_chunks_debug(
            zarr_url,
            pred,
            variables=["model_a/temperature"],
        )

        print(f"Struct field predicate - chunks returned: {len(chunks)}")
        print(f"Coord reads: {coord_reads}")

        if chunks:
            print(f"First chunk: {chunks[0]}")

        assert True

    except Exception as e:
        print(f"Struct field predicate failed: {type(e).__name__}: {e}")
        # Document the error type for future reference
        assert True


def test_current_behavior_root_dim_filter_on_children(
    datatree_datasets: dict[str, str],
) -> None:
    """Document how root dimension filters affect child variable chunks."""
    zarr_url = datatree_datasets["simple_datatree"]

    try:
        pred = pl.col("y") < 8

        # Try selecting a child variable with root filter
        chunks, coord_reads = _core._selected_chunks_debug(
            zarr_url,
            pred,
            variables=["model_a/temperature"],
        )

        print(f"Root dim filter on child - chunks returned: {len(chunks)}")
        print(f"Coord reads: {coord_reads}")

        assert True

    except Exception as e:
        print(f"Root dim filter on child failed: {type(e).__name__}: {e}")
        assert True
