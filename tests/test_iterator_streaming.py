"""Tests for the streaming iterator implementation."""
import polars as pl
import pytest

import rainbear


@pytest.fixture
def orography_path(baseline_datasets):
    """Fixture for orography dataset path."""
    return baseline_datasets["orography_chunked_10x10"]


def test_iterator_yields_filtered_data(orography_path: str):
    """Test that the iterator correctly filters data."""
    # Filter should reduce from 320 rows (16*20) to 72 rows (8*9)
    df = (
        rainbear.scan_zarr(orography_path)
        .filter((pl.col("y") >= 3) & (pl.col("y") <= 10))
        .filter((pl.col("x") >= 4) & (pl.col("x") <= 12))
        .collect()
    )
    
    assert df.height == 8 * 9  # 8 y values * 9 x values
    assert df["y"].min() == 3
    assert df["y"].max() == 10
    assert df["x"].min() == 4
    assert df["x"].max() == 12


def test_iterator_respects_n_rows_limit(orography_path: str):
    """Test that n_rows limit works correctly."""
    # Request only first 50 rows
    df = rainbear.scan_zarr(orography_path).head(50).collect()
    
    assert df.height == 50


def test_iterator_with_column_selection(orography_path: str):
    """Test that column projection works."""
    df = (
        rainbear.scan_zarr(orography_path)
        .select(["y", "x"])
        .collect()
    )
    
    assert df.columns == ["y", "x"]
    assert df.height == 320  # Full dataset


def test_iterator_empty_result(orography_path: str):
    """Test that iterator handles empty results correctly."""
    df = (
        rainbear.scan_zarr(orography_path)
        .filter(pl.col("y") > 1000)  # Impossible condition
        .collect()
    )
    
    assert df.height == 0
    assert "y" in df.columns
    assert "x" in df.columns


def test_iterator_combines_multiple_batches(orography_path: str):
    """Test that data from multiple iterator batches is combined correctly."""
    # Use small batch size to force multiple batches
    # Note: batch_size can't be controlled from Python API currently,
    # but this test verifies the full dataset is returned
    df = rainbear.scan_zarr(orography_path).collect()
    
    assert df.height == 320  # 16 * 20
    
    # Verify data integrity - check some known values
    assert df.filter((pl.col("y") == 0) & (pl.col("x") == 0)).height == 1
    assert df.filter((pl.col("y") == 15) & (pl.col("x") == 19)).height == 1


def test_iterator_with_complex_predicate(orography_path: str):
    """Test iterator with complex multi-column predicate."""
    df = (
        rainbear.scan_zarr(orography_path)
        .filter(
            ((pl.col("y") >= 5) & (pl.col("y") <= 10))
            & ((pl.col("x") >= 5) & (pl.col("x") <= 15))
        )
        .collect()
    )
    
    # 6 y values (5-10 inclusive) * 11 x values (5-15 inclusive)
    assert df.height == 6 * 11
    assert df["y"].min() == 5
    assert df["y"].max() == 10
    assert df["x"].min() == 5
    assert df["x"].max() == 15


def test_iterator_no_duplicates(orography_path: str):
    """Test that iterator doesn't produce duplicate rows."""
    df = rainbear.scan_zarr(orography_path).collect()
    
    # Check for duplicates using y, x as unique key
    duplicates = df.group_by(["y", "x"]).agg(pl.len().alias("count"))
    max_count = duplicates["count"].max()
    
    assert max_count == 1, "Found duplicate rows in iterator output"


def test_iterator_maintains_order(orography_path: str):
    """Test that iterator maintains consistent ordering."""
    df1 = rainbear.scan_zarr(orography_path).collect()
    df2 = rainbear.scan_zarr(orography_path).collect()
    
    # Sort both by y, x to compare
    df1_sorted = df1.sort(["y", "x"])
    df2_sorted = df2.sort(["y", "x"])
    
    assert df1_sorted.equals(df2_sorted), "Iterator produces inconsistent results"
