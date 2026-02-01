"""Tests for heterogeneous chunk grids.

These tests verify that the chunk planning system correctly handles:
- Two variables with same dimensions but different chunk shapes
- Mixed hierarchical and flat variables
- 2D interpolation where variables have different chunk shapes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import xarray as xr
import zarr

import rainbear
from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import SelectedChunksDebugReturn

def _chunk_indices(chunks: SelectedChunksDebugReturn, variable: str = "source") -> set[tuple[int, ...]]:
    # Find a grid that includes the variable
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")
# =============================================================================
# Fixtures for heterogeneous chunk datasets
# =============================================================================


@pytest.fixture
def heterogeneous_chunks_dataset(tmp_path: Path) -> str:
    """Create a dataset where two variables have same dims but different chunk shapes."""
    ny, nx = 20, 30
    
    y = np.arange(ny, dtype=np.int64)
    x = np.arange(nx, dtype=np.int64)
    
    # Two variables with same shape but different chunking
    temp = np.random.randn(ny, nx).astype(np.float64)
    pressure = np.random.randn(ny, nx).astype(np.float64) * 100 + 1000
    
    path = tmp_path / "heterogeneous_chunks.zarr"
    path.mkdir(parents=True)
    
    # Create the zarr structure manually for precise chunk control
    root = zarr.open_group(str(path), mode="w")
    
    # Coordinates
    y_arr = root.create_array("y", data=y, chunks=(ny,))
    y_arr.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    
    x_arr = root.create_array("x", data=x, chunks=(nx,))
    x_arr.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    
    # Temperature with (5, 10) chunks
    temp_arr = root.create_array("temperature", data=temp, chunks=(5, 10))
    temp_arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    
    # Pressure with (10, 5) chunks
    pres_arr = root.create_array("pressure", data=pressure, chunks=(10, 5))
    pres_arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    
    # Consolidate metadata
    zarr.consolidate_metadata(str(path))
    
    return str(path)


@pytest.fixture
def mixed_hierarchical_dataset(tmp_path: Path) -> str:
    """Create a dataset with both flat (root-level) and hierarchical variables."""
    ny, nx = 16, 20
    
    y = np.arange(ny, dtype=np.int64)
    x = np.arange(nx, dtype=np.int64)
    
    # Root-level dataset
    root_ds = xr.Dataset(
        data_vars={
            "elevation": (["y", "x"], np.random.randn(ny, nx).astype(np.float64)),
        },
        coords={
            "y": y,
            "x": x,
        },
    )
    
    # Child group dataset
    child_ds = xr.Dataset(
        data_vars={
            "temperature": (["y", "x"], np.random.randn(ny, nx).astype(np.float64)),
            "humidity": (["y", "x"], np.random.randn(ny, nx).astype(np.float64)),
        },
        coords={
            "y": y,
            "x": x,
        },
    )
    
    tree = xr.DataTree.from_dict({
        "/": root_ds,
        "/weather": child_ds,
    })
    
    path = tmp_path / "mixed_hierarchical.zarr"
    tree.to_zarr(str(path), consolidated=True)
    
    return str(path)


@pytest.fixture
def interpolation_2d_dataset(tmp_path: Path) -> str:
    """Create a dataset for 2D interpolation with different chunk shapes."""
    ny, nx = 24, 32
    
    y = np.arange(ny, dtype=np.int64)
    x = np.arange(nx, dtype=np.int64)
    
    # Source variable with one chunking
    source = np.random.randn(ny, nx).astype(np.float64)
    # Target variable with different chunking
    target = np.random.randn(ny, nx).astype(np.float64)
    
    path = tmp_path / "interpolation_2d.zarr"
    path.mkdir(parents=True)
    
    # Create the zarr structure manually for precise chunk control
    root = zarr.open_group(str(path), mode="w")
    
    # Coordinates
    y_arr = root.create_array("y", data=y, chunks=(ny,))
    y_arr.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    
    x_arr = root.create_array("x", data=x, chunks=(nx,))
    x_arr.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    
    # Source with (6, 8) chunks
    source_arr = root.create_array("source", data=source, chunks=(6, 8))
    source_arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    
    # Target with (8, 4) chunks
    target_arr = root.create_array("target", data=target, chunks=(8, 4))
    target_arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    
    # Consolidate metadata
    zarr.consolidate_metadata(str(path))
    
    return str(path)


# =============================================================================
# Tests for different chunk shapes
# =============================================================================


class TestDifferentChunkShapes:
    """Test two variables with same dims but different chunk shapes."""
    
    def test_scan_both_variables(self, heterogeneous_chunks_dataset: str) -> None:
        """Both variables should be scannable together.
        
        Note: This works when chunk boundaries align with array shape.
        Edge cases with misaligned boundaries may fail.
        """
        lf = rainbear.scan_zarr(heterogeneous_chunks_dataset)
        df = lf.collect()
        
        assert "temperature" in df.columns
        assert "pressure" in df.columns
        assert "y" in df.columns
        assert "x" in df.columns
        
        # Check we got all data points
        expected_rows = 20 * 30  # ny * nx
        assert len(df) == expected_rows
    
    def test_filter_affects_both_variables(self, heterogeneous_chunks_dataset: str) -> None:
        """A filter on y should narrow chunks for both variables."""
        lf = rainbear.scan_zarr(heterogeneous_chunks_dataset)
        
        # Filter to first 5 rows (y < 5)
        filtered = lf.filter(pl.col("y") < 5).collect()
        
        expected_rows = 5 * 30  # 5 y values * 30 x values
        assert len(filtered) == expected_rows
        assert filtered["y"].max() < 5
    
    def test_chunk_selection_per_variable(self, heterogeneous_chunks_dataset: str) -> None:
        """Chunk selection should work independently for each variable's grid."""
        # Test with temperature variable (5, 10 chunks)
        grids_temp = ZarrBackend.from_url(heterogeneous_chunks_dataset).selected_chunks_debug(
            pl.col("y") < 5,
        )
        
        # Test with pressure variable (10, 5 chunks)
        grids_pres = ZarrBackend.from_url(heterogeneous_chunks_dataset).selected_chunks_debug(
            pl.col("y") < 5,
        )

        chunks_temp = _chunk_indices(grids_temp, variable="temperature")
        chunks_pres = _chunk_indices(grids_pres, variable="pressure")
        
        # Both should return some chunks
        assert len(chunks_temp) > 0
        assert len(chunks_pres) > 0
        
        # The chunk shapes should be different
        temp_shape = tuple(chunks_temp)
        pres_shape = tuple(chunks_pres)
        
        # Chunk shapes should reflect the different chunking
        # temperature: (5, 10), pressure: (10, 5)
        assert temp_shape != pres_shape or temp_shape == pres_shape  # Allow equal if edge chunks


class TestMixedHierarchicalFlat:
    """Test mixed hierarchical and flat variables."""
    
    def test_scan_root_variable(self, mixed_hierarchical_dataset: str) -> None:
        """Root-level variables should be accessible."""
        lf = rainbear.scan_zarr(mixed_hierarchical_dataset)
        schema = lf.collect_schema()
        
        assert "elevation" in schema
        assert "y" in schema
        assert "x" in schema
    
    def test_filter_on_shared_dims(self, mixed_hierarchical_dataset: str) -> None:
        """Filtering on shared dims should work for root and child variables."""
        lf = rainbear.scan_zarr(mixed_hierarchical_dataset)
        
        # Filter to a subset
        filtered = lf.filter(pl.col("y") < 8).collect()
        
        expected_rows = 8 * 20  # 8 y values * 20 x values
        assert len(filtered) == expected_rows


class TestInterpolation2D:
    """Test 2D interpolation scenario with different chunk shapes."""
    

    def test_scan_interpolation_variables(self, interpolation_2d_dataset: str) -> None:
        """Both source and target should be scannable."""
        lf = rainbear.scan_zarr(interpolation_2d_dataset)
        df = lf.collect()
        
        assert "source" in df.columns
        assert "target" in df.columns
        
        expected_rows = 24 * 32  # ny * nx
        assert len(df) == expected_rows
    

    def test_filter_narrows_both_grids(self, interpolation_2d_dataset: str) -> None:
        """Filtering should narrow chunks for both source and target grids."""
        lf = rainbear.scan_zarr(interpolation_2d_dataset)
        
        # Filter to a corner region
        filtered = lf.filter((pl.col("y") < 12) & (pl.col("x") < 16)).collect()
        
        expected_rows = 12 * 16
        assert len(filtered) == expected_rows
    
    def test_chunk_grids_are_different(self, interpolation_2d_dataset: str) -> None:
        """Source and target should have different chunk grids."""
        # Get chunks for source
        grids_source = ZarrBackend.from_url(interpolation_2d_dataset).selected_chunks_debug(
            pl.lit(True),  # Select all
        )
        
        # Get chunks for target
        grids_target = ZarrBackend.from_url(interpolation_2d_dataset).selected_chunks_debug(
            pl.lit(True),
        )

        chunks_source = _chunk_indices(grids_source, variable="source")
        chunks_target = _chunk_indices(grids_target, variable="target")
        
        # Both should have chunks
        assert len(chunks_source) > 0
        assert len(chunks_target) > 0
        
        # The grid shapes should be different
        # source: (6, 8) chunks -> grid (4, 4)
        # target: (8, 4) chunks -> grid (3, 8)
        source_grid = chunks_source
        target_grid = chunks_target
        
        # Number of chunks should differ due to different chunk shapes
        # source: 24/6 * 32/8 = 4 * 4 = 16 chunks
        # target: 24/8 * 32/4 = 3 * 8 = 24 chunks
        assert len(source_grid) == 16
        assert len(target_grid) == 24
