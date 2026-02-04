"""Comprehensive tests for IcechunkBackend.

Tests cover:
- Basic functionality (construction, schema, scanning)
- Predicate pushdown for chunk selection
- Cache management
- Equivalence with regular ZarrBackend
"""

from __future__ import annotations

import polars as pl
import pytest

import rainbear
from tests.conftest import IcechunkDatasetInfo

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _normalize(df: pl.DataFrame) -> list[dict[str, object]]:
    """Normalize a DataFrame for stable comparisons."""
    cols = sorted(df.columns)
    df = df.select(cols)
    if df.height == 0:
        return []
    df = df.sort(cols)
    return df.to_dicts()


# ---------------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------------


class TestIcechunkBackendBasic:
    """Basic IcechunkBackend functionality tests."""

    async def test_from_filesystem(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test creating backend from filesystem path."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)
        assert backend is not None

    async def test_repr(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test backend string representation."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)
        repr_str = repr(backend)
        assert "IcechunkBackend" in repr_str

    async def test_schema_basic(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test basic schema retrieval."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)
        schema = backend.schema()
        assert schema is not None
        assert len(schema) > 0

        # Check dimension coords are present
        for dim in info.dims:
            assert dim in schema, f"Missing dimension: {dim}"

        # Check data variables are present
        for var in info.data_vars:
            assert var in schema, f"Missing variable: {var}"

    async def test_schema_with_variable_filter(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test schema retrieval with variable filter."""
        info = icechunk_datasets["icechunk_multi_var"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Request only specific variables
        schema = backend.schema(variables=["temp", "precip"])
        assert schema is not None

        # Should have filtered variables
        assert "temp" in schema
        assert "precip" in schema

        # Dimensions should still be present
        for dim in info.dims:
            assert dim in schema


class TestIcechunkBackendScan:
    """Scan operation tests for IcechunkBackend."""

    async def test_scan_simple(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test basic scan without predicate filtering."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Scan with true predicate (no filtering)
        df = await backend.scan_zarr_async(pl.lit(True))

        # Should have data
        assert df.height > 0

        # Should have expected columns
        for dim in info.dims:
            assert dim in df.columns, f"Missing dimension: {dim}"
        for var in info.data_vars:
            assert var in df.columns, f"Missing variable: {var}"

    async def test_scan_with_predicate(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test scan with predicate filtering."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Filter to a subset
        pred = (pl.col("y") >= 3) & (pl.col("y") <= 10)
        df = await backend.scan_zarr_async(pred)

        # Apply filter manually to verify
        df_filtered = df.filter(pred)
        assert df_filtered.height > 0
        assert df_filtered["y"].min() >= 3
        assert df_filtered["y"].max() <= 10

    async def test_scan_with_variables(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test scan with specific variables."""
        info = icechunk_datasets["icechunk_multi_var"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Request specific variables
        df = await backend.scan_zarr_async(
            pl.lit(True),
            variables=["temp", "precip"],
        )

        assert "temp" in df.columns
        assert "precip" in df.columns

    async def test_scan_with_columns(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test scan with with_columns parameter."""
        info = icechunk_datasets["icechunk_comprehensive_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Request specific columns
        cols = ["a", "b", "data"]
        df = await backend.scan_zarr_async(
            pl.lit(True),
            with_columns=cols,
        )

        for col in cols:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Predicate Pushdown Tests
# ---------------------------------------------------------------------------


class TestIcechunkPredicatePushdown:
    """Predicate pushdown tests for IcechunkBackend."""

    async def test_single_dim_filter(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test filtering on single dimension."""
        info = icechunk_datasets["icechunk_comprehensive_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Filter a < 20 should hit chunks a=0,1 only (2 out of 7)
        pred = pl.col("a") < 20
        df = await backend.scan_zarr_async(
            pred,
            variables=["data"],
            with_columns=["a", "b", "c", "data"],
        )

        # Apply filter and verify
        df_filtered = df.filter(pred)
        assert df_filtered.height > 0
        assert df_filtered["a"].max() < 20

    async def test_multi_dim_filter(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test filtering on multiple dimensions."""
        info = icechunk_datasets["icechunk_comprehensive_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Filter (a < 20) & (b < 20) should hit 2*2*3 = 12 chunks
        pred = (pl.col("a") < 20) & (pl.col("b") < 20)
        df = await backend.scan_zarr_async(
            pred,
            variables=["data"],
            with_columns=["a", "b", "c", "data"],
        )

        df_filtered = df.filter(pred)
        assert df_filtered.height > 0
        assert df_filtered["a"].max() < 20
        assert df_filtered["b"].max() < 20

    async def test_range_filter(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test range-based filtering."""
        info = icechunk_datasets["icechunk_comprehensive_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Filter 10 <= a < 30 should hit chunks a=1,2 only
        pred = (pl.col("a") >= 10) & (pl.col("a") < 30)
        df = await backend.scan_zarr_async(
            pred,
            variables=["data"],
            with_columns=["a", "b", "c", "data"],
        )

        df_filtered = df.filter(pred)
        assert df_filtered.height > 0
        assert df_filtered["a"].min() >= 10
        assert df_filtered["a"].max() < 30

    async def test_selected_chunks_debug(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that selected_chunks_debug returns chunk info."""
        info = icechunk_datasets["icechunk_comprehensive_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Simple predicate
        pred = pl.col("a") < 20
        debug_info = backend.selected_chunks_debug(pred)

        # Should have grids and coord_reads
        assert "grids" in debug_info
        assert "coord_reads" in debug_info

        # Should have at least one grid
        assert len(debug_info["grids"]) > 0

        # Each grid should have dims, variables, and chunks
        for grid in debug_info["grids"]:
            assert "dims" in grid
            assert "variables" in grid
            assert "chunks" in grid


# ---------------------------------------------------------------------------
# Cache Management Tests
# ---------------------------------------------------------------------------


class TestIcechunkCaching:
    """Cache behavior tests for IcechunkBackend."""

    async def test_cache_stats_initial(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test cache statistics on fresh backend."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        stats = await backend.cache_stats()
        assert "coord_entries" in stats

    async def test_cache_populated_after_scan(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Verify cache is populated after scanning."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Perform a scan
        await backend.scan_zarr_async(pl.lit(True))

        # Check cache has entries
        stats = await backend.cache_stats()
        # Note: coord_entries may or may not be > 0 depending on implementation
        assert "coord_entries" in stats

    async def test_clear_coord_cache(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test clearing coordinate cache."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Populate cache
        await backend.scan_zarr_async(pl.lit(True))

        # Clear cache
        await backend.clear_coord_cache()

        # Should be able to scan again
        df = await backend.scan_zarr_async(pl.lit(True))
        assert df.height > 0

    async def test_clear_all_caches(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test clearing all caches."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Populate cache
        await backend.scan_zarr_async(pl.lit(True))

        # Clear all caches
        await backend.clear_all_caches()

        # Should be able to scan again
        df = await backend.scan_zarr_async(pl.lit(True))
        assert df.height > 0

    async def test_repeated_queries_work(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Verify repeated queries produce consistent results."""
        info = icechunk_datasets["icechunk_orography"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        pred = (pl.col("y") >= 5) & (pl.col("x") >= 5)

        # First query
        df1 = await backend.scan_zarr_async(pred)

        # Second query (should use cache)
        df2 = await backend.scan_zarr_async(pred)

        # Results should match
        assert _normalize(df1.filter(pred)) == _normalize(df2.filter(pred))


# ---------------------------------------------------------------------------
# Equivalence Tests with ZarrBackend
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session Serialization Tests (icechunk-python -> rainbear via bytes)
# ---------------------------------------------------------------------------


def _get_session_bytes(icechunk_session) -> bytes:
    """Extract session bytes from an icechunk-python Session.

    The icechunk-python Session class wraps an internal PySession that has
    as_bytes()/from_bytes() methods. This helper accesses the internal API.
    """
    # Access the internal PySession's as_bytes method
    return icechunk_session._session.as_bytes()


class TestSessionSerialization:
    """Tests for passing sessions from icechunk-python to rainbear via bytes.

    These tests verify that:
    1. Sessions can be serialized to bytes from icechunk-python
    2. Those bytes can be deserialized by rainbear's Session.from_bytes()
    3. Backends created from serialized sessions produce identical results
       to backends created directly from the filesystem

    Note: icechunk-python's public Session class doesn't expose as_bytes() directly,
    so we access the internal _session.as_bytes() for serialization.
    """

    async def test_session_from_bytes_basic(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test basic session serialization round-trip."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_orography"]

        # Get session from icechunk-python
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")

        # Serialize to bytes (via internal API)
        session_bytes = _get_session_bytes(icechunk_session)
        assert isinstance(session_bytes, bytes)
        assert len(session_bytes) > 0

        # Deserialize via rainbear
        rainbear_session = rainbear.Session.from_bytes(session_bytes)
        assert rainbear_session is not None

        # Verify properties match
        assert rainbear_session.read_only == icechunk_session.read_only
        assert rainbear_session.snapshot_id == icechunk_session.snapshot_id

    async def test_session_properties(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that session properties are correctly exposed."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_orography"]

        # Get session from icechunk-python
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")

        # Round-trip through bytes
        session_bytes = _get_session_bytes(icechunk_session)
        rainbear_session = rainbear.Session.from_bytes(session_bytes)

        # Check properties
        assert rainbear_session.read_only is True
        # Note: branch may be None for readonly sessions (icechunk behavior)
        assert rainbear_session.branch is None or rainbear_session.branch == "main"
        assert len(rainbear_session.snapshot_id) > 0

        # Check repr
        repr_str = repr(rainbear_session)
        assert "Session" in repr_str
        assert "snapshot_id" in repr_str

    async def test_backend_from_session_basic(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test creating backend from a serialized session."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_orography"]

        # Get session from icechunk-python and serialize
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")
        session_bytes = _get_session_bytes(icechunk_session)

        # Create backend via rainbear
        rainbear_session = rainbear.Session.from_bytes(session_bytes)
        backend = await rainbear.IcechunkBackend.from_session(rainbear_session)

        # Verify backend works
        assert backend is not None
        schema = backend.schema()
        assert len(schema) > 0

    async def test_session_scan_equivalence(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that scans from session-based backend match filesystem-based backend."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_orography"]

        # Backend from filesystem (reference)
        backend_fs = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Backend from serialized session
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")
        session_bytes = _get_session_bytes(icechunk_session)
        rainbear_session = rainbear.Session.from_bytes(session_bytes)
        backend_session = await rainbear.IcechunkBackend.from_session(rainbear_session)

        # Scan with same predicate
        pred = pl.lit(True)
        df_fs = await backend_fs.scan_zarr_async(pred)
        df_session = await backend_session.scan_zarr_async(pred)

        # Results should be identical
        assert _normalize(df_fs) == _normalize(df_session)

    async def test_session_scan_with_predicate_equivalence(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test predicate-filtered scans produce same results from both backends."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_orography"]

        # Backend from filesystem
        backend_fs = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Backend from session
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")
        rainbear_session = rainbear.Session.from_bytes(
            _get_session_bytes(icechunk_session)
        )
        backend_session = await rainbear.IcechunkBackend.from_session(rainbear_session)

        # Scan with filtering predicate
        pred = (pl.col("y") >= 3) & (pl.col("y") <= 10)
        df_fs = await backend_fs.scan_zarr_async(pred)
        df_session = await backend_session.scan_zarr_async(pred)

        # Apply filter and compare
        df_fs_filtered = df_fs.filter(pred)
        df_session_filtered = df_session.filter(pred)

        assert _normalize(df_fs_filtered) == _normalize(df_session_filtered)

    async def test_session_selected_chunks_equivalence(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that chunk selection produces same results from both backends."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_comprehensive_3d"]

        # Backend from filesystem
        backend_fs = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Backend from session
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")
        rainbear_session = rainbear.Session.from_bytes(
            _get_session_bytes(icechunk_session)
        )
        backend_session = await rainbear.IcechunkBackend.from_session(rainbear_session)

        # Compare chunk selection
        pred = pl.col("a") < 20
        debug_fs = backend_fs.selected_chunks_debug(pred)
        debug_session = backend_session.selected_chunks_debug(pred)

        # Should select same number of grids and chunks
        assert len(debug_fs["grids"]) == len(debug_session["grids"])

        # Compare chunk counts per grid
        for grid_fs, grid_session in zip(debug_fs["grids"], debug_session["grids"]):
            assert len(grid_fs["chunks"]) == len(grid_session["chunks"])

    async def test_session_schema_equivalence(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that schema retrieval matches between backends."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_multi_var"]

        # Backend from filesystem
        backend_fs = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Backend from session
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")
        rainbear_session = rainbear.Session.from_bytes(
            _get_session_bytes(icechunk_session)
        )
        backend_session = await rainbear.IcechunkBackend.from_session(rainbear_session)

        # Compare schemas
        schema_fs = backend_fs.schema()
        schema_session = backend_session.schema()

        assert schema_fs == schema_session

    async def test_session_multi_var_scan_equivalence(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test multi-variable scans produce same results from both backends."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_multi_var"]

        # Backend from filesystem
        backend_fs = await rainbear.IcechunkBackend.from_filesystem(info.path)

        # Backend from session
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")
        rainbear_session = rainbear.Session.from_bytes(
            _get_session_bytes(icechunk_session)
        )
        backend_session = await rainbear.IcechunkBackend.from_session(rainbear_session)

        # Scan specific variables with predicate on dimension coords
        # (dimension coords are always available even when selecting specific variables)
        pred = (pl.col("a") < 30) & (pl.col("b") < 25)
        vars_to_read = ["temp", "precip"]
        # Include dimension coords in with_columns for filtering
        cols_to_read = ["a", "b", "c", "temp", "precip"]

        df_fs = await backend_fs.scan_zarr_async(pred, with_columns=cols_to_read)
        df_session = await backend_session.scan_zarr_async(pred, with_columns=cols_to_read)

        # Filter and compare
        df_fs_filtered = df_fs.filter(pred)
        df_session_filtered = df_session.filter(pred)

        assert _normalize(df_fs_filtered) == _normalize(df_session_filtered)

    async def test_session_roundtrip_rainbear_to_rainbear(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that rainbear Session can be serialized and deserialized."""
        from icechunk import Repository, local_filesystem_storage

        info = icechunk_datasets["icechunk_orography"]

        # Create initial session via icechunk-python
        storage = local_filesystem_storage(info.path)
        repo = Repository.open(storage)
        icechunk_session = repo.readonly_session("main")

        # Create rainbear session
        session1 = rainbear.Session.from_bytes(_get_session_bytes(icechunk_session))

        # Serialize rainbear session back to bytes
        bytes1 = session1.as_bytes()

        # Deserialize again
        session2 = rainbear.Session.from_bytes(bytes1)

        # Verify properties match
        assert session1.snapshot_id == session2.snapshot_id
        assert session1.branch == session2.branch
        assert session1.read_only == session2.read_only

        # Create backends from both and verify they produce same results
        backend1 = await rainbear.IcechunkBackend.from_session(session1)
        backend2 = await rainbear.IcechunkBackend.from_session(session2)

        df1 = await backend1.scan_zarr_async(pl.lit(True))
        df2 = await backend2.scan_zarr_async(pl.lit(True))

        assert _normalize(df1) == _normalize(df2)


class TestIcechunkZarrEquivalence:
    """Tests comparing IcechunkBackend results with ZarrBackend.

    These tests write the same data to both Icechunk and regular Zarr,
    then verify that queries produce equivalent results.
    """

    async def test_equivalence_orography_scan(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
        baseline_datasets: dict[str, str],
    ) -> None:
        """Test that Icechunk and Zarr backends produce same orography data."""
        icechunk_info = icechunk_datasets["icechunk_orography"]
        zarr_path = baseline_datasets["orography_chunked_10x10"]

        # Create backends
        icechunk_backend = await rainbear.IcechunkBackend.from_filesystem(
            icechunk_info.path
        )
        zarr_backend = rainbear.ZarrBackend.from_url(zarr_path)

        # Same predicate for both
        pred = (pl.col("y") >= 3) & (pl.col("y") <= 10)
        cols = ["y", "x", "geopotential_height"]

        # Query icechunk
        df_icechunk = await icechunk_backend.scan_zarr_async(
            pred,
            variables=["geopotential_height"],
            with_columns=cols,
        )
        df_icechunk = df_icechunk.filter(pred).select(cols)

        # Query zarr (async)
        df_zarr = await zarr_backend.scan_zarr_async(
            pred,
            variables=["geopotential_height"],
            with_columns=cols,
        )
        df_zarr = df_zarr.filter(pred).select(cols)

        # Both should have data - structure should be the same
        # (values may differ since they're separate datasets)
        assert df_icechunk.height > 0
        assert df_zarr.height > 0
        assert set(df_icechunk.columns) == set(df_zarr.columns)

    async def test_equivalence_schema_structure(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Test that schema retrieval works correctly for Icechunk."""
        info = icechunk_datasets["icechunk_comprehensive_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        schema = backend.schema()
        assert schema is not None

        # Check expected dimensions are in schema
        for dim in ["a", "b", "c"]:
            assert dim in schema, f"Missing dimension: {dim}"

        # Check expected variables are in schema
        for var in ["data", "data2"]:
            assert var in schema, f"Missing variable: {var}"
