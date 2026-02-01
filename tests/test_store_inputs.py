"""Tests for store-based inputs (ObjectStore instances instead of URL strings).

This test module verifies that users can pass ObjectStore instances to scan_zarr
and scan_zarr_async, in addition to URL strings.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

import rainbear


class TestStoreModuleExports:
    """Test that store and exceptions modules are properly exported."""

    def test_store_module_available(self) -> None:
        """Verify rainbear.store module is accessible."""
        assert hasattr(rainbear, "store")

    def test_exceptions_module_available(self) -> None:
        """Verify rainbear.exceptions module is accessible."""
        assert hasattr(rainbear, "exceptions")

    def test_s3_store_available(self) -> None:
        """Verify S3Store is available in the store module."""
        assert hasattr(rainbear.store, "S3Store")

    def test_gcs_store_available(self) -> None:
        """Verify GCSStore is available in the store module."""
        assert hasattr(rainbear.store, "GCSStore")

    def test_azure_store_available(self) -> None:
        """Verify AzureStore is available in the store module."""
        assert hasattr(rainbear.store, "AzureStore")

    def test_http_store_available(self) -> None:
        """Verify HTTPStore is available in the store module."""
        assert hasattr(rainbear.store, "HTTPStore")

    def test_local_store_available(self) -> None:
        """Verify LocalStore is available in the store module."""
        assert hasattr(rainbear.store, "LocalStore")

    def test_memory_store_available(self) -> None:
        """Verify MemoryStore is available in the store module."""
        assert hasattr(rainbear.store, "MemoryStore")


class TestUrlStringInput:
    """Regression tests for URL string inputs (existing behavior)."""

    def test_url_string_relative_path(self, baseline_datasets: dict[str, str]) -> None:
        """Verify URL string with relative path works."""
        zarr_url = baseline_datasets["orography_chunked_10x10"]
        lf = rainbear.scan_zarr(zarr_url)
        df = lf.collect()

        assert df.height == 16 * 20
        assert "geopotential_height" in df.columns

    def test_url_string_absolute_path(self, baseline_datasets: dict[str, str]) -> None:
        """Verify URL string with absolute path works."""
        zarr_url = os.path.abspath(baseline_datasets["orography_chunked_10x10"])
        lf = rainbear.scan_zarr(zarr_url)
        df = lf.collect()

        assert df.height == 16 * 20
        assert "geopotential_height" in df.columns


class TestLocalStoreInput:
    """Tests for LocalStore inputs."""

    def test_local_store_with_absolute_prefix(self, baseline_datasets: dict[str, str]) -> None:
        """Verify LocalStore with absolute path prefix works."""
        local_store = rainbear.store.LocalStore()
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])

        lf = rainbear.scan_zarr(local_store, prefix=abs_path, variables=["geopotential_height"])
        df = lf.collect()

        assert df.height == 16 * 20
        assert "geopotential_height" in df.columns

    def test_local_store_multiple_variables(self, baseline_datasets: dict[str, str]) -> None:
        """Verify LocalStore works with multiple variables."""
        local_store = rainbear.store.LocalStore()
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])

        lf = rainbear.scan_zarr(
            local_store,
            prefix=abs_path,
            variables=["geopotential_height", "latitude", "longitude"],
        )
        df = lf.collect()

        assert df.height == 16 * 20
        assert set(df.columns) == {"y", "x", "geopotential_height", "latitude", "longitude"}

    def test_local_store_with_filter(self, baseline_datasets: dict[str, str]) -> None:
        """Verify LocalStore works with predicate pushdown."""
        local_store = rainbear.store.LocalStore()
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])

        lf = rainbear.scan_zarr(local_store, prefix=abs_path, variables=["geopotential_height"])
        lf = lf.filter((pl.col("y") >= 3) & (pl.col("y") <= 10))
        df = lf.collect()

        assert df.filter(pl.col("y") < 3).is_empty()
        assert df.filter(pl.col("y") > 10).is_empty()


class TestStoreInputEquivalence:
    """Tests that URL string and ObjectStore inputs produce equivalent results."""

    def test_url_vs_local_store_equivalence(self, baseline_datasets: dict[str, str]) -> None:
        """Verify URL string and LocalStore produce identical results."""
        zarr_path = baseline_datasets["orography_chunked_10x10"]
        abs_path = os.path.abspath(zarr_path)

        # Using URL string
        df_url = rainbear.scan_zarr(zarr_path).collect()

        # Using LocalStore
        local_store = rainbear.store.LocalStore()
        df_store = rainbear.scan_zarr(
            local_store, prefix=abs_path, variables=["geopotential_height"]
        ).collect()

        # Results should be identical
        assert df_url.height == df_store.height
        assert df_url.columns == df_store.columns
        # Sort to ensure order-independent comparison
        df_url_sorted = df_url.sort(["y", "x"])
        df_store_sorted = df_store.sort(["y", "x"])
        assert df_url_sorted.equals(df_store_sorted)

    def test_url_vs_local_store_with_filter_equivalence(
        self, baseline_datasets: dict[str, str]
    ) -> None:
        """Verify URL string and LocalStore produce identical results with filters."""
        zarr_path = baseline_datasets["orography_chunked_10x10"]
        abs_path = os.path.abspath(zarr_path)

        filter_expr = (pl.col("y") >= 3) & (pl.col("y") <= 10) & (pl.col("x") >= 4) & (pl.col("x") <= 12)

        # Using URL string
        df_url = (
            rainbear.scan_zarr(zarr_path)
            .filter(filter_expr)
            .collect()
        )

        # Using LocalStore
        local_store = rainbear.store.LocalStore()
        df_store = (
            rainbear.scan_zarr(local_store, prefix=abs_path, variables=["geopotential_height"])
            .filter(filter_expr)
            .collect()
        )

        # Results should be identical
        assert df_url.height == df_store.height
        df_url_sorted = df_url.sort(["y", "x"])
        df_store_sorted = df_store.sort(["y", "x"])
        assert df_url_sorted.equals(df_store_sorted)


class TestZarrSourceDirectUsage:
    """Tests for direct ZarrBackendSync usage with ObjectStore inputs."""

    def test_zarr_source_with_url_string(self, baseline_datasets: dict[str, str]) -> None:
        """Verify ZarrBackendSync works with URL string."""
        zarr_path = baseline_datasets["orography_chunked_10x10"]

        src = rainbear.ZarrBackendSync.from_url(
            url=zarr_path
        )
        df = src.scan_zarr_sync(    
            variables=["geopotential_height"])

        assert df.height == 16 * 20
        assert "geopotential_height" in df.columns

    def test_zarr_source_with_local_store(self, baseline_datasets: dict[str, str]) -> None:
        """Verify ZarrBackendSync works with LocalStore."""
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])
        local_store = rainbear.store.LocalStore()

        src = rainbear.ZarrBackendSync.from_store(
            store=local_store,
            prefix=abs_path,
        )
        df = src.scan_zarr_sync(
            variables=["geopotential_height"],
        )

        assert df.height == 16 * 20
        assert "geopotential_height" in df.columns


class TestExternalObstoreCompatibility:
    """Tests for external obstore package compatibility.

    These tests verify that if users have the obstore package installed,
    they can pass obstore instances to rainbear functions.
    """

    @pytest.fixture
    def obstore_available(self) -> bool:
        """Check if obstore package is available."""
        try:
            import obstore  # noqa: F401
            return True
        except ImportError:
            return False

    def test_external_obstore_local_store(
        self, baseline_datasets: dict[str, str], obstore_available: bool
    ) -> None:
        """Verify external obstore LocalStore works if available."""
        if not obstore_available:
            pytest.skip("obstore package not installed")

        import obstore

        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])
        external_store = obstore.store.LocalStore()

        lf = rainbear.scan_zarr(
            external_store, prefix=abs_path, variables=["geopotential_height"]
        )
        df = lf.collect()

        assert df.height == 16 * 20
        assert "geopotential_height" in df.columns


class TestStoreInputEdgeCases:
    """Edge case tests for store inputs."""

    def test_prefix_with_leading_slash(self, baseline_datasets: dict[str, str]) -> None:
        """Verify prefix with leading slash works correctly."""
        local_store = rainbear.store.LocalStore()
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])
        # Add leading slash explicitly
        prefixed_path = "/" + abs_path.lstrip("/")

        lf = rainbear.scan_zarr(
            local_store, prefix=prefixed_path, variables=["geopotential_height"]
        )
        df = lf.collect()

        assert df.height == 16 * 20

    def test_prefix_without_leading_slash(self, baseline_datasets: dict[str, str]) -> None:
        """Verify prefix without leading slash works correctly."""
        local_store = rainbear.store.LocalStore()
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])
        # Remove leading slash if present
        prefixed_path = abs_path.lstrip("/")

        lf = rainbear.scan_zarr(
            local_store, prefix=prefixed_path, variables=["geopotential_height"]
        )
        df = lf.collect()

        assert df.height == 16 * 20

    def test_none_prefix_with_url_string(self, baseline_datasets: dict[str, str]) -> None:
        """Verify None prefix with URL string works (prefix ignored for URLs)."""
        zarr_path = baseline_datasets["orography_chunked_10x10"]

        lf = rainbear.scan_zarr(zarr_path, prefix=None, variables=["geopotential_height"])
        df = lf.collect()

        assert df.height == 16 * 20

    def test_max_chunks_with_store_input(self, baseline_datasets: dict[str, str]) -> None:
        """Verify max_chunks_to_read works with ObjectStore inputs."""
        local_store = rainbear.store.LocalStore()
        abs_path = os.path.abspath(baseline_datasets["orography_chunked_10x10"])

        lf = rainbear.scan_zarr(
            local_store,
            prefix=abs_path,
            variables=["geopotential_height"],
            max_chunks_to_read=1,
        )

        with pytest.raises(pl.exceptions.ComputeError, match="max_chunks_to_read"):
            lf.collect()
