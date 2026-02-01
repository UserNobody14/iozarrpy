"""Tests for ZarrBackend caching functionality."""

import asyncio

import polars as pl
import pytest

import rainbear


class TestZarrBackendCreation:
    """Tests for backend creation methods."""

    def test_from_url_local_path(self, baseline_datasets: dict[str, str]):
        """Test creating a backend from a local path."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)
        assert backend is not None

    def test_from_url_with_max_cache(self, baseline_datasets: dict[str, str]):
        """Test creating a backend with cache limit."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(
            zarr_url,
            max_cache_entries=100,
        )
        assert backend is not None

    def test_backend_repr(self, baseline_datasets: dict[str, str]):
        """Test backend string representation."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)
        repr_str = repr(backend)
        assert "ZarrBackend" in repr_str
        assert "root=" in repr_str


class TestZarrBackendSchema:
    """Tests for schema retrieval."""

    def test_schema_all_variables(self, baseline_datasets: dict[str, str]):
        """Test getting schema for all variables."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)
        schema = backend.schema()
        assert schema is not None
        # Should have dimensions and data variables
        assert len(schema) > 0

    def test_schema_specific_variables(self, baseline_datasets: dict[str, str]):
        """Test getting schema for specific variables."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)
        schema = backend.schema(variables=["geopotential_height"])
        assert schema is not None


class TestZarrBackendAsyncScan:
    """Tests for async scanning with caching."""

    @pytest.mark.asyncio
    async def test_scan_zarr_async_basic(self, baseline_datasets: dict[str, str]):
        """Test basic async scan."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        # Simple scan with trivially true predicate
        df = await backend.scan_zarr_async(pl.lit(True))

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_scan_zarr_async_with_predicate(
        self, baseline_datasets: dict[str, str]
    ):
        """Test async scan with a filtering predicate."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        # Get all data first to know what to filter
        df_all = await backend.scan_zarr_async(pl.lit(True))

        if "y" in df_all.columns and len(df_all) > 0:
            mid_y = df_all["y"].mean()
            df_filtered = await backend.scan_zarr_async(pl.col("y") > mid_y)
            # Filtered should have fewer or equal rows
            assert len(df_filtered) <= len(df_all)

    @pytest.mark.asyncio
    async def test_cache_persistence_across_scans(
        self, baseline_datasets: dict[str, str]
    ):
        """Test that cache persists across multiple scans."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

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
    async def test_scan_with_variables(self, baseline_datasets: dict[str, str]):
        """Test async scan with specific variables."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        # Scan with specific variable
        df = await backend.scan_zarr_async(
            pl.lit(True),
            variables=["geopotential_height"],
        )

        assert isinstance(df, pl.DataFrame)
        # Should have geopotential_height column
        assert "geopotential_height" in df.columns


class TestZarrBackendCacheManagement:
    """Tests for cache management."""

    @pytest.mark.asyncio
    async def test_cache_stats(self, baseline_datasets: dict[str, str]):
        """Test getting cache statistics."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        # Initially cache should be empty or minimal
        stats = await backend.cache_stats()
        assert "coord_entries" in stats
        assert "has_metadata" in stats

    @pytest.mark.asyncio
    async def test_clear_coord_cache(self, baseline_datasets: dict[str, str]):
        """Test clearing coordinate cache."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        # Trigger some caching
        await backend.scan_zarr_async(pl.lit(True))

        # Clear cache
        await backend.clear_coord_cache()

        # Stats should show fewer entries (metadata still cached)
        stats = await backend.cache_stats()
        assert stats["coord_entries"] == 0

    @pytest.mark.asyncio
    async def test_clear_all_caches(self, baseline_datasets: dict[str, str]):
        """Test clearing all caches."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

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
    async def test_with_sharded_dataset(self, baseline_datasets: dict[str, str]):
        """Test backend with sharded dataset."""
        zarr_url = baseline_datasets["orography_sharded_small"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        df = await backend.scan_zarr_async(pl.lit(True))
        assert isinstance(df, pl.DataFrame)

    @pytest.mark.asyncio
    async def test_with_multi_var(self, baseline_datasets: dict[str, str]):
        """Test backend with multi-variable dataset."""
        zarr_url = baseline_datasets["orography_chunked_5x5"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        df = await backend.scan_zarr_async(pl.lit(True))
        assert isinstance(df, pl.DataFrame)

    @pytest.mark.asyncio
    async def test_with_different_chunks(self, baseline_datasets: dict[str, str]):
        """Test backend with different chunk sizes."""
        zarr_url = baseline_datasets["orography_sharded_small"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        df = await backend.scan_zarr_async(pl.lit(True))
        assert isinstance(df, pl.DataFrame)


class TestZarrBackendConcurrency:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_scans_same_backend(
        self, baseline_datasets: dict[str, str]
    ):
        """Test multiple concurrent scans on the same backend."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        backend = rainbear.ZarrBackend.from_url(zarr_url)

        # Use simple predicate for concurrent tests
        predicate = pl.col("y") > 8

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
