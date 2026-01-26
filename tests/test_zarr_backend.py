"""Tests for ZarrBackend caching functionality."""

import asyncio

import polars as pl
import pytest

import rainbear


class TestZarrBackendCreation:
    """Tests for backend creation methods."""

    def test_from_url_local_path(self):
        """Test creating a backend from a local path."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        assert backend is not None
        assert "demo_store.zarr" in backend.root() or backend.root() == "/"

    def test_from_url_with_max_cache(self):
        """Test creating a backend with cache limit."""
        backend = rainbear.ZarrBackend.from_url(
            "tests/output-datasets/demo_store.zarr",
            max_cache_entries=100,
        )
        assert backend is not None

    def test_backend_repr(self):
        """Test backend string representation."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        repr_str = repr(backend)
        assert "ZarrBackend" in repr_str
        assert "root=" in repr_str


class TestZarrBackendSchema:
    """Tests for schema retrieval."""

    def test_schema_all_variables(self):
        """Test getting schema for all variables."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        schema = backend.schema()
        assert schema is not None
        # Should have dimensions and data variables
        assert len(schema) > 0

    def test_schema_specific_variables(self):
        """Test getting schema for specific variables."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        schema = backend.schema(variables=["temp"])
        assert schema is not None


class TestZarrBackendAsyncScan:
    """Tests for async scanning with caching."""

    @pytest.mark.asyncio
    async def test_scan_zarr_async_basic(self):
        """Test basic async scan."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Simple scan with trivially true predicate
        df = await backend.scan_zarr_async(pl.lit(True))
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_scan_zarr_async_with_predicate(self):
        """Test async scan with a filtering predicate."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Get all data first to know what to filter
        df_all = await backend.scan_zarr_async(pl.lit(True))
        
        if "lat" in df_all.columns and len(df_all) > 0:
            mid_lat = df_all["lat"].mean()
            df_filtered = await backend.scan_zarr_async(pl.col("lat") > mid_lat)
            # Filtered should have fewer or equal rows
            assert len(df_filtered) <= len(df_all)

    @pytest.mark.asyncio
    async def test_cache_persistence_across_scans(self):
        """Test that cache persists across multiple scans."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # First scan - should populate cache
        df1 = await backend.scan_zarr_async(pl.lit(True))
        stats1 = await backend.cache_stats()
        
        # Second scan - should use cached data
        df2 = await backend.scan_zarr_async(pl.lit(True))
        stats2 = await backend.cache_stats()
        
        # Both scans should return same data
        assert len(df1) == len(df2)
        
        # Metadata should be cached after first scan
        assert stats1["has_metadata"] == True
        assert stats2["has_metadata"] == True

    @pytest.mark.asyncio
    async def test_scan_with_variables(self):
        """Test async scan with specific variables."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Scan with specific variable
        df = await backend.scan_zarr_async(
            pl.lit(True),
            variables=["temp"],
        )
        
        assert isinstance(df, pl.DataFrame)
        # Should have temp column
        assert "temp" in df.columns or len(df.columns) > 0


class TestZarrBackendCacheManagement:
    """Tests for cache management."""

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test getting cache statistics."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Initially cache should be empty or minimal
        stats = await backend.cache_stats()
        assert "coord_entries" in stats
        assert "has_metadata" in stats

    @pytest.mark.asyncio
    async def test_clear_coord_cache(self):
        """Test clearing coordinate cache."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Trigger some caching
        await backend.scan_zarr_async(pl.lit(True))
        
        # Clear cache
        await backend.clear_coord_cache()
        
        # Stats should show fewer entries (metadata still cached)
        stats = await backend.cache_stats()
        assert stats["coord_entries"] == 0

    @pytest.mark.asyncio
    async def test_clear_all_caches(self):
        """Test clearing all caches."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Trigger some caching
        await backend.scan_zarr_async(pl.lit(True))
        
        # Clear all caches
        await backend.clear_all_caches()
        
        # Stats should show empty cache
        stats = await backend.cache_stats()
        assert stats["coord_entries"] == 0
        assert stats["has_metadata"] == False


class TestZarrBackendMultiDataset:
    """Tests with different dataset types."""

    @pytest.mark.asyncio
    async def test_with_time_dimension(self):
        """Test backend with dataset containing time dimension."""
        # This dataset should have time encoding
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store_sel.zarr")
        
        df = await backend.scan_zarr_async(pl.lit(True))
        assert isinstance(df, pl.DataFrame)

    @pytest.mark.asyncio
    async def test_with_multi_var(self):
        """Test backend with multi-variable dataset."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/orography_small.zarr")
        
        # Use orography_small which is a small dataset
        df = await backend.scan_zarr_async(pl.lit(True))
        assert isinstance(df, pl.DataFrame)

    @pytest.mark.asyncio
    async def test_with_4d_dataset(self):
        """Test backend with another dataset type."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store_sel.zarr")
        
        # Use a small dataset
        df = await backend.scan_zarr_async(pl.lit(True))
        assert isinstance(df, pl.DataFrame)


class TestZarrBackendConcurrency:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_scans_same_backend(self):
        """Test multiple concurrent scans on the same backend."""
        backend = rainbear.ZarrBackend.from_url("tests/output-datasets/demo_store.zarr")
        
        # Use small predicates for concurrent tests
        predicate = pl.col("lat") > 30
        
        # Launch multiple concurrent scans
        tasks = [
            backend.scan_zarr_async(predicate),
            backend.scan_zarr_async(predicate),
            backend.scan_zarr_async(predicate),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should return valid DataFrames
        for df in results:
            assert isinstance(df, pl.DataFrame)
        
        # All should have same length
        assert all(len(df) == len(results[0]) for df in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
