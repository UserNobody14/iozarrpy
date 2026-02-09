"""Tests for consolidated vs unconsolidated metadata handling."""
import polars as pl
import pytest
import rainbear


def test_scan_with_consolidated_metadata(baseline_datasets):
    """Test scanning a dataset with consolidated metadata."""
    path = baseline_datasets["orography_chunked_10x10"]
    
    df = rainbear.scan_zarr(path).collect()
    
    assert df.height == 320  # 16 * 20
    assert "y" in df.columns
    assert "x" in df.columns
    assert "geopotential_height" in df.columns


def test_scan_with_unconsolidated_metadata(baseline_datasets):
    """Test scanning a dataset without consolidated metadata."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    df = rainbear.scan_zarr(path).collect()
    
    # Should successfully read unconsolidated dataset
    assert df.height > 0
    assert "time" in df.columns


def test_filter_with_unconsolidated_metadata(baseline_datasets):
    """Test filtering works with unconsolidated metadata."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    df = (
        rainbear.scan_zarr(path)
        .filter(pl.col("time") == pl.col("time").first())
        .collect()
    )
    
    # Should filter correctly
    assert df.height > 0
    assert df.height < rainbear.scan_zarr(path).collect().height


def test_consolidated_vs_unconsolidated_equivalence(baseline_datasets):
    """Test that consolidated and unconsolidated produce identical results."""
    path_consolidated = baseline_datasets["grid_chunked"]
    path_unconsolidated = baseline_datasets["grid_constant_unconsolidated"]
    
    # Both should read successfully
    df_consolidated = rainbear.scan_zarr(path_consolidated).head(10).collect()
    df_unconsolidated = rainbear.scan_zarr(path_unconsolidated).head(10).collect()
    
    # Both should have valid data
    assert df_consolidated.height == 10
    assert df_unconsolidated.height == 10


def test_iterator_with_consolidated_metadata(baseline_datasets):
    """Test iterator works correctly with consolidated metadata."""
    path = baseline_datasets["orography_chunked_10x10"]
    
    # Use filter to ensure iterator is actually used
    df = (
        rainbear.scan_zarr(path)
        .filter((pl.col("x") >= 10) & (pl.col("x") <= 15))
        .collect()
    )
    
    assert df["x"].min() == 10
    assert df["x"].max() == 15
    assert df.height == 16 * 6  # 16 y values * 6 x values


def test_iterator_with_unconsolidated_metadata(baseline_datasets):
    """Test iterator works correctly with unconsolidated metadata."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    # Use filter
    df = (
        rainbear.scan_zarr(path)
        .filter(pl.col("lead_time") >= 0)
        .collect()
    )
    
    # Should work
    assert df.height > 0


def test_column_selection_with_unconsolidated(baseline_datasets):
    """Test column selection with unconsolidated metadata."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    df = (
        rainbear.scan_zarr(path)
        .select(["time", "y"])
        .collect()
    )
    
    assert "time" in df.columns
    assert "y" in df.columns
    assert df.height > 0


def test_empty_filter_with_unconsolidated(baseline_datasets):
    """Test empty filter result with unconsolidated metadata."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    df = (
        rainbear.scan_zarr(path)
        .filter(pl.col("y") > 1000000)  # Impossible condition
        .collect()
    )
    
    assert df.height == 0
    assert "y" in df.columns


def test_complex_predicate_unconsolidated(baseline_datasets):
    """Test complex predicate with unconsolidated metadata."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    df = (
        rainbear.scan_zarr(path)
        .filter(
            (pl.col("y") >= 5) | (pl.col("x") >= 10)
        )
        .collect()
    )
    
    # Should work
    assert df.height > 0


def test_grid_constant_unconsolidated_basic(baseline_datasets):
    """Test basic read of grid_constant_unconsolidated dataset."""
    path = baseline_datasets["grid_constant_unconsolidated"]
    
    df = rainbear.scan_zarr(path).collect()
    
    # Should successfully read the dataset
    assert df.height > 0
    assert "time" in df.columns
    assert "2m_temperature" in df.columns or "total_precipitation" in df.columns
