"""Benchmarks comparing caching backend vs non-caching approaches.

This benchmark tests the advantage of ZarrBackend's coordinate caching
by running multiple queries on the same dataset. The caching backend
should show significant speedup on repeated queries.

Compares:
- xarray (opened once, reused)
- rainbear.scan_zarr (LazyFrame, reopens each query)
- rainbear.scan_zarr_async (reopens each query)
- rainbear.ZarrBackend.scan_zarr_async (caches coords across queries)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta

import polars as pl
import pytest
import xarray as xr

import rainbear

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_dataset_path() -> str:
    """Get the benchmark dataset path.
    
    Uses RAINBEAR_BENCH_DATASET env var if set, otherwise uses a local test dataset.
    """
    custom = os.environ.get("RAINBEAR_BENCH_DATASET")
    if custom:
        return custom
    # Fall back to a local test dataset
    return "tests/output-datasets/orography_chunked_10x10.zarr"


@pytest.fixture(scope="module")
def xarray_dataset(bench_dataset_path: str):
    """Open xarray dataset once, reuse for all iterations."""
    ds = xr.open_zarr(bench_dataset_path, chunks=None)
    yield ds
    ds.close()


@pytest.fixture(scope="module")
def zarr_backend(bench_dataset_path: str) -> rainbear.ZarrBackend:
    """Create caching backend once, reuse for all iterations."""
    return rainbear.ZarrBackend.from_url(bench_dataset_path)


# ---------------------------------------------------------------------------
# Query predicates
# ---------------------------------------------------------------------------


def _pred_subset() -> pl.Expr:
    """A restrictive predicate that selects a small subset."""
    return (
        (pl.col("y") >= 3)
        & (pl.col("y") <= 8)
        & (pl.col("x") >= 4)
        & (pl.col("x") <= 10)
    )


# ---------------------------------------------------------------------------
# Implementation functions
# ---------------------------------------------------------------------------


def impl_xarray_reused(ds: xr.Dataset) -> pl.DataFrame:
    """Query using pre-opened xarray dataset."""
    pdf = ds.sel(
        y=slice(3, 8),
        x=slice(4, 10),
    )[["geopotential_height"]].to_dataframe().reset_index()
    df = pl.from_pandas(pdf)
    return df.filter(_pred_subset()).select(["y", "x", "geopotential_height"])


def impl_xarray_fresh(path: str) -> pl.DataFrame:
    """Query by opening fresh xarray dataset each time."""
    ds = xr.open_zarr(path, consolidated=False)
    try:
        pdf = ds.sel(
            y=slice(3, 8),
            x=slice(4, 10),
        )[["geopotential_height"]].to_dataframe().reset_index()
    finally:
        ds.close()
    df = pl.from_pandas(pdf)
    return df.filter(_pred_subset()).select(["y", "x", "geopotential_height"])


def impl_scan_zarr(path: str) -> pl.DataFrame:
    """Query using rainbear.scan_zarr (LazyFrame)."""
    pred = _pred_subset()
    return (
        rainbear.scan_zarr(path)
        .filter(pred)
        .select(["y", "x", "geopotential_height"])
        .collect()
    )


def impl_scan_zarr_async(path: str) -> pl.DataFrame:
    """Query using rainbear.scan_zarr_async (no caching)."""
    pred = _pred_subset()

    async def _run() -> pl.DataFrame:
        df = await rainbear.scan_zarr_async(
            path,
            pred,
            variables=["geopotential_height"],
            max_concurrency=8,
            with_columns=["y", "x", "geopotential_height"],
        )
        return df.filter(pred).select(["y", "x", "geopotential_height"])

    return asyncio.run(_run())


def impl_backend_cached(backend: rainbear.ZarrBackend) -> pl.DataFrame:
    """Query using caching ZarrBackend."""
    pred = _pred_subset()

    async def _run() -> pl.DataFrame:
        df = await backend.scan_zarr_async(
            pred,
            variables=["geopotential_height"],
            max_concurrency=8,
            with_columns=["y", "x", "geopotential_height"],
        )
        return df.filter(pred).select(["y", "x", "geopotential_height"])

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Single-query benchmarks (baseline)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="single_query")
def test_bench_single_xarray_reused(benchmark, xarray_dataset) -> None:
    """Single query with pre-opened xarray."""
    out = benchmark(impl_xarray_reused, xarray_dataset)
    assert "geopotential_height" in out.columns


@pytest.mark.benchmark(group="single_query")
def test_bench_single_xarray_fresh(benchmark, bench_dataset_path: str) -> None:
    """Single query opening fresh xarray each time."""
    out = benchmark(impl_xarray_fresh, bench_dataset_path)
    assert "geopotential_height" in out.columns


@pytest.mark.benchmark(group="single_query")
def test_bench_single_scan_zarr(benchmark, bench_dataset_path: str) -> None:
    """Single query using scan_zarr."""
    out = benchmark(impl_scan_zarr, bench_dataset_path)
    assert "geopotential_height" in out.columns


@pytest.mark.benchmark(group="single_query")
def test_bench_single_scan_zarr_async(benchmark, bench_dataset_path: str) -> None:
    """Single query using scan_zarr_async."""
    out = benchmark(impl_scan_zarr_async, bench_dataset_path)
    assert "geopotential_height" in out.columns


@pytest.mark.benchmark(group="single_query")
def test_bench_single_backend_cached(benchmark, zarr_backend: rainbear.ZarrBackend) -> None:
    """Single query using caching backend."""
    out = benchmark(impl_backend_cached, zarr_backend)
    assert "geopotential_height" in out.columns


# ---------------------------------------------------------------------------
# Multi-query benchmarks (where caching shines)
# ---------------------------------------------------------------------------


def _multi_query_xarray_reused(ds: xr.Dataset, n: int = 5) -> list[pl.DataFrame]:
    """Run N queries on pre-opened xarray."""
    results = []
    for _ in range(n):
        results.append(impl_xarray_reused(ds))
    return results


def _multi_query_scan_zarr(path: str, n: int = 5) -> list[pl.DataFrame]:
    """Run N queries using scan_zarr (reopens each time)."""
    results = []
    for _ in range(n):
        results.append(impl_scan_zarr(path))
    return results


def _multi_query_scan_zarr_async(path: str, n: int = 5) -> list[pl.DataFrame]:
    """Run N queries using scan_zarr_async (reopens each time)."""
    results = []
    for _ in range(n):
        results.append(impl_scan_zarr_async(path))
    return results


def _multi_query_backend_cached(backend: rainbear.ZarrBackend, n: int = 5) -> list[pl.DataFrame]:
    """Run N queries using caching backend (coords cached after first query)."""
    results = []
    for _ in range(n):
        results.append(impl_backend_cached(backend))
    return results


@pytest.mark.benchmark(group="multi_query_5x")
def test_bench_multi_xarray_reused(benchmark, xarray_dataset) -> None:
    """5 queries with pre-opened xarray."""
    out = benchmark(_multi_query_xarray_reused, xarray_dataset)
    assert len(out) == 5


@pytest.mark.benchmark(group="multi_query_5x")
def test_bench_multi_scan_zarr(benchmark, bench_dataset_path: str) -> None:
    """5 queries using scan_zarr."""
    out = benchmark(_multi_query_scan_zarr, bench_dataset_path)
    assert len(out) == 5


@pytest.mark.benchmark(group="multi_query_5x")
def test_bench_multi_scan_zarr_async(benchmark, bench_dataset_path: str) -> None:
    """5 queries using scan_zarr_async."""
    out = benchmark(_multi_query_scan_zarr_async, bench_dataset_path)
    assert len(out) == 5


@pytest.mark.benchmark(group="multi_query_5x")
def test_bench_multi_backend_cached(benchmark, zarr_backend: rainbear.ZarrBackend) -> None:
    """5 queries using caching backend - should show speedup from caching."""
    out = benchmark(_multi_query_backend_cached, zarr_backend)
    assert len(out) == 5


# ---------------------------------------------------------------------------
# Concurrent query benchmarks
# ---------------------------------------------------------------------------


def _concurrent_queries_backend(backend: rainbear.ZarrBackend, n: int = 5) -> list[pl.DataFrame]:
    """Run N concurrent queries on caching backend."""
    pred = _pred_subset()

    async def _run() -> list[pl.DataFrame]:
        async def single_query() -> pl.DataFrame:
            df = await backend.scan_zarr_async(
                pred,
                variables=["geopotential_height"],
                max_concurrency=4,
                with_columns=["y", "x", "geopotential_height"],
            )
            return df.filter(pred).select(["y", "x", "geopotential_height"])

        tasks = [single_query() for _ in range(n)]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())


@pytest.mark.benchmark(group="concurrent_5x")
def test_bench_concurrent_backend_cached(benchmark, zarr_backend: rainbear.ZarrBackend) -> None:
    """5 concurrent queries using caching backend."""
    out = benchmark(_concurrent_queries_backend, zarr_backend)
    assert len(out) == 5
    for df in out:
        assert "geopotential_height" in df.columns


# ---------------------------------------------------------------------------
# Remote dataset benchmarks (opt-in)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def remote_dataset_path() -> str | None:
    """Get remote dataset path from env, or skip."""
    return os.environ.get("RAINBEAR_REMOTE_MEM")


@pytest.fixture(scope="module")
def remote_backend(remote_dataset_path: str | None):
    """Create caching backend for remote dataset."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
    return rainbear.ZarrBackend.from_url(remote_dataset_path)


@pytest.fixture(scope="module")
def remote_xarray_dataset(remote_dataset_path: str | None):
    """Open remote xarray dataset once, reuse for all iterations."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
    ds = xr.open_zarr(remote_dataset_path, chunks=None)
    yield ds
    ds.close()


def _remote_pred() -> pl.Expr:
    """Predicate for remote dataset (single point in space-time)."""
    return (
        (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0))
        & (pl.col("lead_time") == timedelta(hours=1))
        & (pl.col("y") == 13)
        & (pl.col("x") == 13)
    )


def impl_remote_xarray_fresh(path: str) -> pl.DataFrame:
    """Query remote dataset by opening fresh xarray each time."""
    ds = xr.open_zarr(path, chunks=None)
    try:
        # Select single point in space-time (use lists to keep dimensions)
        pdf = ds.sel(
            time=[datetime(2025, 12, 30, 0, 0, 0)],
            lead_time=[timedelta(hours=1)],
            y=[13.0],
            x=[13.0],
        )[["80m_wind_speed"]].to_dataframe()
    finally:
        ds.close()
    return pl.from_pandas(pdf)


def impl_remote_xarray_reused(ds: xr.Dataset) -> pl.DataFrame:
    """Query remote dataset using pre-opened xarray."""
    pdf = ds.sel(
        time=[datetime(2025, 12, 30, 0, 0, 0)],
        lead_time=[timedelta(hours=1)],
        y=[13.0],
        x=[13.0],
    )[["80m_wind_speed"]].to_dataframe()
    return pl.from_pandas(pdf)


def impl_remote_scan_zarr(path: str) -> pl.DataFrame:
    """Query remote dataset using scan_zarr (LazyFrame)."""
    pred = _remote_pred()
    return (
        rainbear.scan_zarr(path)
        .filter(pred)
        .select(["time", "lead_time", "y", "x", "80m_wind_speed"])
        .collect()
    )


def impl_remote_scan_zarr_async(path: str) -> pl.DataFrame:
    """Query remote dataset using scan_zarr_async."""
    pred = _remote_pred()

    async def _run() -> pl.DataFrame:
        df = await rainbear.scan_zarr_async(
            path,
            pred,
            variables=["80m_wind_speed"],
            with_columns=["time", "lead_time", "x", "y", "80m_wind_speed"],
        )
        return df

    return asyncio.run(_run())


def impl_remote_backend_cached(backend: rainbear.ZarrBackend) -> pl.DataFrame:
    """Query remote dataset using caching backend."""
    pred = _remote_pred()

    async def _run() -> pl.DataFrame:
        df = await backend.scan_zarr_async(
            pred,
            variables=["80m_wind_speed"],
            with_columns=["time", "lead_time", "x", "y", "80m_wind_speed"],
        )
        return df

    return asyncio.run(_run())


# Single query benchmarks - remote

@pytest.mark.benchmark(group="remote_single")
def test_bench_remote_xarray_fresh(benchmark, remote_dataset_path: str | None) -> None:
    """Single remote query opening fresh xarray each time."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
    out = benchmark(impl_remote_xarray_fresh, remote_dataset_path)
    assert len(out) >= 1


@pytest.mark.benchmark(group="remote_single")
def test_bench_remote_xarray_reused(benchmark, remote_xarray_dataset) -> None:
    """Single remote query with pre-opened xarray."""
    out = benchmark(impl_remote_xarray_reused, remote_xarray_dataset)
    assert len(out) >= 1


# @pytest.mark.benchmark(group="remote_single")
# def test_bench_remote_scan_zarr(benchmark, remote_dataset_path: str | None) -> None:
#     """Single remote query using scan_zarr."""
#     if not remote_dataset_path:
#         pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
#     out = benchmark(impl_remote_scan_zarr, remote_dataset_path)
#     assert len(out) >= 1


@pytest.mark.benchmark(group="remote_single")
def test_bench_remote_scan_zarr_async(benchmark, remote_dataset_path: str | None) -> None:
    """Single remote query using scan_zarr_async."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
    out = benchmark(impl_remote_scan_zarr_async, remote_dataset_path)
    assert out.height >= 1


@pytest.mark.benchmark(group="remote_single")
def test_bench_remote_backend_cached(benchmark, remote_backend) -> None:
    """Single remote query using caching backend."""
    out = benchmark(impl_remote_backend_cached, remote_backend)
    assert out.height >= 1


# Multi-query benchmarks - remote (where caching really matters)

def _remote_multi_query_xarray_fresh(path: str, n: int = 3) -> list[pl.DataFrame]:
    """Run N queries on remote dataset, opening fresh xarray each time."""
    results = []
    for _ in range(n):
        results.append(impl_remote_xarray_fresh(path))
    return results


def _remote_multi_query_xarray_reused(ds: xr.Dataset, n: int = 3) -> list[pl.DataFrame]:
    """Run N queries on remote dataset with pre-opened xarray."""
    results = []
    for _ in range(n):
        results.append(impl_remote_xarray_reused(ds))
    return results


def _remote_multi_query_scan_zarr(path: str, n: int = 3) -> list[pl.DataFrame]:
    """Run N queries on remote dataset using scan_zarr."""
    results = []
    for _ in range(n):
        results.append(impl_remote_scan_zarr(path))
    return results


def _remote_multi_query_async(path: str, n: int = 3) -> list[pl.DataFrame]:
    """Run N queries on remote dataset without caching."""
    results = []
    for _ in range(n):
        results.append(impl_remote_scan_zarr_async(path))
    return results


def _remote_multi_query_backend(backend: rainbear.ZarrBackend, n: int = 3) -> list[pl.DataFrame]:
    """Run N queries on remote dataset with caching."""
    results = []
    for _ in range(n):
        results.append(impl_remote_backend_cached(backend))
    return results


@pytest.mark.benchmark(group="remote_multi_3x")
def test_bench_remote_multi_xarray_fresh(benchmark, remote_dataset_path: str | None) -> None:
    """3 remote queries opening fresh xarray each time (worst case)."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
    out = benchmark(_remote_multi_query_xarray_fresh, remote_dataset_path)
    assert len(out) == 3


@pytest.mark.benchmark(group="remote_multi_3x")
def test_bench_remote_multi_xarray_reused(benchmark, remote_xarray_dataset) -> None:
    """3 remote queries with pre-opened xarray (xarray's best case)."""
    out = benchmark(_remote_multi_query_xarray_reused, remote_xarray_dataset)
    assert len(out) == 3


# @pytest.mark.benchmark(group="remote_multi_3x")
# def test_bench_remote_multi_scan_zarr(benchmark, remote_dataset_path: str | None) -> None:
#     """3 remote queries using scan_zarr."""
#     if not remote_dataset_path:
#         pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
#     out = benchmark(_remote_multi_query_scan_zarr, remote_dataset_path)
#     assert len(out) == 3


@pytest.mark.benchmark(group="remote_multi_3x")
def test_bench_remote_multi_scan_zarr_async(benchmark, remote_dataset_path: str | None) -> None:
    """3 remote queries using scan_zarr_async (no caching)."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_MEM for remote benchmarks")
    out = benchmark(_remote_multi_query_async, remote_dataset_path)
    assert len(out) == 3


@pytest.mark.benchmark(group="remote_multi_3x")
def test_bench_remote_multi_backend_cached(benchmark, remote_backend) -> None:
    """3 remote queries using caching backend - should be faster on 2nd+ query."""
    out = benchmark(_remote_multi_query_backend, remote_backend)
    assert len(out) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
