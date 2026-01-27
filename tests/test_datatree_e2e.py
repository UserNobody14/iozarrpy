"""End-to-end tests for zarr hierarchies (xarray DataTrees).

These tests document the expected behavior for representing zarr group hierarchies
as nested Polars struct columns. Tests are expected to fail until hierarchical
support is implemented in rainbear.

DataTree nodes should map to Polars struct fields:
- Root variables become top-level columns
- Child groups become struct columns containing their variables
- Deeply nested groups become nested structs

Example expected schema for a simple tree:
    {
        "y": Int64,
        "x": Int64,
        "surface": Float64,
        "model_a": Struct({
            "temperature": Float64,
            "pressure": Float64,
        }),
        "model_b": Struct({
            "temperature": Float64,
            "humidity": Float64,
        }),
    }
"""

from __future__ import annotations

import polars as pl

import rainbear

# =============================================================================
# Basic DataTree Scan Tests
# =============================================================================


def test_simple_datatree_scan(datatree_datasets: dict[str, str]) -> None:
    """Test scanning a simple 2-level DataTree."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Attempt to scan - currently expected to fail or only see root variables
    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Expected: root dims + root vars + child groups as structs
    expected_columns = {"y", "x", "surface", "model_a", "model_b"}
    assert set(df.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {set(df.columns)}"
    )

    # Check that model_a is a struct column
    assert df["model_a"].dtype == pl.Struct, (
        f"Expected model_a to be Struct, got {df['model_a'].dtype}"
    )

    # Check struct field access
    model_a_temp = df["model_a"].struct.field("temperature")
    assert model_a_temp.dtype == pl.Float64


def test_nested_groups_as_structs(datatree_datasets: dict[str, str]) -> None:
    """Verify that child groups become struct columns with proper field types."""
    zarr_url = datatree_datasets["simple_datatree"]

    lf = rainbear.scan_zarr(zarr_url)
    schema = lf.collect_schema()

    # model_a should be a struct with temperature and pressure fields
    model_a_type = schema.get("model_a")
    assert model_a_type is not None, "model_a should be in schema"
    assert isinstance(model_a_type, pl.Struct), f"Expected Struct, got {model_a_type}"

    # Check the struct fields
    model_a_fields = {f.name: f.dtype for f in model_a_type.fields}
    assert "temperature" in model_a_fields
    assert "pressure" in model_a_fields
    assert model_a_fields["temperature"] == pl.Float64
    assert model_a_fields["pressure"] == pl.Float64

    # model_b should have temperature and humidity (different from model_a)
    model_b_type = schema.get("model_b")
    assert model_b_type is not None, "model_b should be in schema"
    model_b_fields = {f.name: f.dtype for f in model_b_type.fields}
    assert "temperature" in model_b_fields
    assert "humidity" in model_b_fields


def test_datatree_variable_selection(datatree_datasets: dict[str, str]) -> None:
    """Test selecting specific variables from nested groups."""
    zarr_url = datatree_datasets["simple_datatree"]

    # Select only model_a/temperature using struct field syntax
    lf = rainbear.scan_zarr(zarr_url)
    df = lf.select(
        "y",
        "x",
        pl.col("model_a").struct.field("temperature").alias("model_a_temp"),
    ).collect()

    assert df.columns == ["y", "x", "model_a_temp"]
    assert df["model_a_temp"].dtype == pl.Float64


# =============================================================================
# Deep Hierarchy Tests
# =============================================================================


def test_deep_hierarchy_traversal(datatree_datasets: dict[str, str]) -> None:
    """Test 4+ level tree with nested struct representation."""
    zarr_url = datatree_datasets["deep_tree_d4"]

    lf = rainbear.scan_zarr(zarr_url)
    schema = lf.collect_schema()

    # Root should have root_var and level_1 struct
    assert "root_var" in schema
    assert "level_1" in schema

    # level_1 should be a struct containing var_1 and level_2
    level_1_type = schema["level_1"]
    assert isinstance(level_1_type, pl.Struct)

    # Navigate through the nested structure
    df = lf.collect()

    # Access deeply nested variable: level_1.level_2.level_3.level_4.var_4
    # This requires chaining struct field access
    deep_var = (
        df["level_1"]
        .struct.field("level_2")
        .struct.field("level_3")
        .struct.field("level_4")
        .struct.field("var_4")
    )
    assert deep_var.dtype == pl.Float64
    assert len(deep_var) == df.height


def test_deep_hierarchy_intermediate_access(datatree_datasets: dict[str, str]) -> None:
    """Test accessing intermediate levels in deep hierarchy."""
    zarr_url = datatree_datasets["deep_tree_d4"]

    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Access level_2 variable (intermediate depth)
    level_2_var = df["level_1"].struct.field("level_2").struct.field("var_2")
    assert level_2_var.dtype == pl.Float64

    # Access level_3 variable
    level_3_var = (
        df["level_1"].struct.field("level_2").struct.field("level_3").struct.field("var_3")
    )
    assert level_3_var.dtype == pl.Float64


# =============================================================================
# Heterogeneous Schema Tests
# =============================================================================


def test_heterogeneous_schemas(datatree_datasets: dict[str, str]) -> None:
    """Test nodes with different variable schemas and dimensions."""
    zarr_url = datatree_datasets["heterogeneous_tree"]

    lf = rainbear.scan_zarr(zarr_url)
    schema = lf.collect_schema()

    # Should have surface_2d, atmosphere_3d, and timeseries_1d as struct columns
    assert "surface_2d" in schema
    assert "atmosphere_3d" in schema
    assert "timeseries_1d" in schema

    # surface_2d should have 2D variables
    surface_type = schema["surface_2d"]
    assert isinstance(surface_type, pl.Struct)
    surface_fields = {f.name for f in surface_type.fields}
    assert surface_fields == {"elevation", "land_mask"}

    # atmosphere_3d has additional z dimension - how is this represented?
    # This is a design question: does z become part of the struct or a separate dim?
    atmos_type = schema["atmosphere_3d"]
    assert isinstance(atmos_type, pl.Struct)


def test_heterogeneous_different_dims(datatree_datasets: dict[str, str]) -> None:
    """Test that nodes with different dimensions are handled correctly."""
    zarr_url = datatree_datasets["heterogeneous_tree"]

    # The timeseries_1d node has a completely different dimension (station_id)
    # This tests how rainbear handles non-uniform dimension structures
    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Access the timeseries data
    timeseries = df["timeseries_1d"]
    assert timeseries.dtype == pl.Struct

    # The measurement field should be accessible
    measurement = timeseries.struct.field("measurement")
    assert measurement is not None


# =============================================================================
# Ensemble/Wide Tree Tests
# =============================================================================


def test_ensemble_tree_members(datatree_datasets: dict[str, str]) -> None:
    """Test ensemble forecast structure with multiple member children."""
    zarr_url = datatree_datasets["ensemble_tree"]

    lf = rainbear.scan_zarr(zarr_url)
    schema = lf.collect_schema()

    # Root should have orography and time/y/x dims
    assert "orography" in schema
    assert "time" in schema
    assert "y" in schema
    assert "x" in schema

    # Should have member_0 through member_4 as struct columns
    for i in range(5):
        member_key = f"member_{i}"
        assert member_key in schema, f"Missing {member_key}"
        member_type = schema[member_key]
        assert isinstance(member_type, pl.Struct), f"{member_key} should be Struct"


def test_ensemble_member_access(datatree_datasets: dict[str, str]) -> None:
    """Test accessing variables from specific ensemble members."""
    zarr_url = datatree_datasets["ensemble_tree"]

    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Access temperature from member_0
    member_0_temp = df["member_0"].struct.field("temperature")
    assert member_0_temp.dtype == pl.Float64

    # Access precipitation from member_2
    member_2_precip = df["member_2"].struct.field("precipitation")
    assert member_2_precip.dtype == pl.Float64


def test_wide_tree_many_children(datatree_datasets: dict[str, str]) -> None:
    """Test tree with many sibling children (horizontal scaling)."""
    zarr_url = datatree_datasets["wide_tree_n10"]

    lf = rainbear.scan_zarr(zarr_url)
    schema = lf.collect_schema()

    # Should have child_0 through child_9
    for i in range(10):
        child_key = f"child_{i}"
        assert child_key in schema, f"Missing {child_key}"
        child_type = schema[child_key]
        assert isinstance(child_type, pl.Struct)

        # Each child should have a 'data' field
        child_fields = {f.name for f in child_type.fields}
        assert "data" in child_fields


# =============================================================================
# Unconsolidated Zarr Tests
# =============================================================================


def test_datatree_unconsolidated(datatree_datasets: dict[str, str]) -> None:
    """Test scanning unconsolidated hierarchical zarr store."""
    zarr_url = datatree_datasets["simple_datatree_unconsolidated"]

    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Should work the same as consolidated
    expected_columns = {"y", "x", "surface", "model_a", "model_b"}
    assert set(df.columns) == expected_columns


# =============================================================================
# Row Count Verification Tests
# =============================================================================


def test_datatree_row_count(datatree_datasets: dict[str, str]) -> None:
    """Verify correct row count for hierarchical data."""
    zarr_url = datatree_datasets["simple_datatree"]

    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Simple datatree has nx=20, ny=16
    expected_rows = 20 * 16
    assert df.height == expected_rows, f"Expected {expected_rows} rows, got {df.height}"


def test_ensemble_tree_row_count(datatree_datasets: dict[str, str]) -> None:
    """Verify correct row count for ensemble tree with time dimension."""
    zarr_url = datatree_datasets["ensemble_tree"]

    lf = rainbear.scan_zarr(zarr_url)
    df = lf.collect()

    # Ensemble tree has nx=30, ny=24, nt=6
    # Row count depends on how time is handled with structs
    # If fully flattened: 30 * 24 * 6 = 4320
    expected_rows = 30 * 24 * 6
    assert df.height == expected_rows, f"Expected {expected_rows} rows, got {df.height}"


# =============================================================================
# Tests to Document Current Behavior
# =============================================================================


def test_current_behavior_simple_datatree(datatree_datasets: dict[str, str]) -> None:
    """Document what currently happens when scanning a DataTree.

    This test does NOT use xfail - it captures the actual current behavior
    so we can track changes as hierarchical support is added.
    """
    zarr_url = datatree_datasets["simple_datatree"]

    try:
        lf = rainbear.scan_zarr(zarr_url)
        df = lf.collect()

        # Record what columns we actually get
        actual_columns = set(df.columns)
        print(f"Current columns: {actual_columns}")
        print(f"Schema: {df.schema}")

        # Does it see any child group variables?
        sees_model_a = "model_a" in actual_columns
        sees_nested_temp = any("temperature" in c for c in actual_columns)

        print(f"Sees model_a as column: {sees_model_a}")
        print(f"Sees any temperature column: {sees_nested_temp}")

        # This assertion always passes - we're just documenting
        assert True

    except Exception as e:
        # Document the error if scanning fails entirely
        print(f"Scanning hierarchical zarr failed with: {type(e).__name__}: {e}")
        # Don't fail the test - we're documenting behavior
        assert True


def test_current_behavior_deep_tree(datatree_datasets: dict[str, str]) -> None:
    """Document what currently happens when scanning a deep DataTree."""
    zarr_url = datatree_datasets["deep_tree_d4"]

    try:
        lf = rainbear.scan_zarr(zarr_url)
        df = lf.collect()

        print(f"Deep tree columns: {set(df.columns)}")
        print(f"Deep tree schema: {df.schema}")
        print(f"Deep tree row count: {df.height}")

        # Check if any nested level variables are visible
        for level in range(1, 5):
            var_name = f"var_{level}"
            level_name = f"level_{level}"
            has_var = var_name in df.columns
            has_level = level_name in df.columns
            print(f"Level {level}: has var={has_var}, has level struct={has_level}")

        assert True

    except Exception as e:
        print(f"Deep tree scan failed: {type(e).__name__}: {e}")
        assert True
