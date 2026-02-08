"""Benchmarks for novel query performance.

This benchmark tests how different approaches handle truly novel queries -
where each iteration uses different coordinate ranges (different lat/lon bounds).
This is a more realistic simulation of production use cases where cached
identical queries are rare.

Key differences from test_benchmark_backend_caching.py:
- Each benchmark iteration uses DIFFERENT coordinate bounds
- Measures cold-start + planning overhead, not query caching benefits
- Pre-generates N random predicates to ensure fair comparison

Dataset used: comprehensive_4d.zarr
- Dimensions: time (6), lead_time (10), y (70), x (40)
- Chunk shape: (2, 2, 10, 10) -> 3x5x7x4 = 420 total chunks
- Coordinate ranges: y in [0,69], x in [0,39]

Compares:
- xarray (fresh open each query)
- xarray (pre-opened, reused handle)
- rainbear.scan_zarr (sync LazyFrame)
- rainbear.scan_zarr_async (async, no backend caching)
- rainbear.ZarrBackend.scan_zarr_async (async, with coord caching)
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timedelta
from typing import NamedTuple

import polars as pl
import pytest
import xarray as xr

import rainbear

# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

# Dataset coordinate bounds (from comprehensive_4d.zarr)
Y_MIN, Y_MAX = 0, 69
X_MIN, X_MAX = 0, 39
TIME_COUNT = 6  # 6 time steps
LEAD_COUNT = 10  # 10 lead times


class QueryBounds(NamedTuple):
    """Bounds for a single query."""

    y_min: int
    y_max: int
    x_min: int
    x_max: int
    time_idx: int
    lead_idx: int


def generate_random_queries(n: int, seed: int = 42) -> list[QueryBounds]:
    """Generate N random query bounds.

    Each query selects:
    - A random rectangular region in y/x space (5-20 points per dim)
    - A specific time index
    - A specific lead_time index

    Using a fixed seed ensures reproducibility and fair comparison
    across different approaches.
    """
    rng = random.Random(seed)
    queries = []

    for _ in range(n):
        # Random y range (5-20 points)
        y_size = rng.randint(5, 20)
        y_start = rng.randint(Y_MIN, Y_MAX - y_size)
        y_end = y_start + y_size

        # Random x range (5-15 points)
        x_size = rng.randint(5, 15)
        x_start = rng.randint(X_MIN, X_MAX - x_size)
        x_end = x_start + x_size

        # Random time and lead_time
        time_idx = rng.randint(0, TIME_COUNT - 1)
        lead_idx = rng.randint(0, LEAD_COUNT - 1)

        queries.append(
            QueryBounds(
                y_min=y_start,
                y_max=y_end,
                x_min=x_start,
                x_max=x_end,
                time_idx=time_idx,
                lead_idx=lead_idx,
            )
        )

    return queries


def query_to_polars_pred(q: QueryBounds, base_time: datetime) -> pl.Expr:
    """Convert QueryBounds to a polars predicate expression."""
    target_time = base_time + timedelta(hours=q.time_idx * 6)
    target_lead = timedelta(hours=q.lead_idx)

    return (
        (pl.col("y") >= q.y_min)
        & (pl.col("y") <= q.y_max)
        & (pl.col("x") >= q.x_min)
        & (pl.col("x") <= q.x_max)
        & (pl.col("time") == target_time)
        & (pl.col("lead_time") == target_lead)
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_dataset_path() -> str:
    """Get the benchmark dataset path."""
    custom = os.environ.get("RAINBEAR_BENCH_DATASET_4D")
    if custom:
        return custom
    return "tests/output-datasets/comprehensive_4d.zarr"


@pytest.fixture(scope="module")
def base_time() -> datetime:
    """Base time for the dataset (from zarr_generators.py)."""
    return datetime(2024, 1, 1)


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


@pytest.fixture(scope="module")
def zarr_backend_sync(bench_dataset_path: str) -> rainbear.ZarrBackendSync:
    """Create sync caching backend once, reuse for all iterations."""
    return rainbear.ZarrBackendSync.from_url(bench_dataset_path)


@pytest.fixture(scope="module")
def query_set_small() -> list[QueryBounds]:
    """Pre-generated set of 10 random queries."""
    return generate_random_queries(10, seed=12345)


@pytest.fixture(scope="module")
def query_set_medium() -> list[QueryBounds]:
    """Pre-generated set of 50 random queries."""
    return generate_random_queries(50, seed=54321)


_columns = ["y", "x", "time", "lead_time", "temperature"]


# ---------------------------------------------------------------------------
# Implementation functions
# ---------------------------------------------------------------------------


def impl_xarray_fresh(path: str, q: QueryBounds, base_time: datetime) -> int:
    """Query by opening fresh xarray dataset each time."""
    target_time = base_time + timedelta(hours=q.time_idx * 6)
    target_lead = timedelta(hours=q.lead_idx)

    ds = xr.open_zarr(path, chunks=None)
    try:
        result_fresh = ds.sel(
            time=target_time,
            lead_time=target_lead,
            y=slice(q.y_min, q.y_max),
            x=slice(q.x_min, q.x_max),
        )["temperature"]
        result = result_fresh.to_dataframe().reset_index()
    finally:
        ds.close()
    return len(result)


def impl_xarray_reused(ds: xr.Dataset, q: QueryBounds, base_time: datetime) -> int:
    """Query using pre-opened xarray dataset."""
    target_time = base_time + timedelta(hours=q.time_idx * 6)
    target_lead = timedelta(hours=q.lead_idx)

    result = ds.sel(
        time=target_time,
        lead_time=target_lead,
        y=slice(q.y_min, q.y_max),
        x=slice(q.x_min, q.x_max),
    )["temperature"]
    result = result.to_dataframe().reset_index()
    return len(result)


def impl_scan_zarr(path: str, q: QueryBounds, base_time: datetime) -> int:
    """Query using rainbear.scan_zarr (sync LazyFrame)."""
    pred = query_to_polars_pred(q, base_time)
    df = (
        rainbear.scan_zarr(path)
        .filter(pred)
        .select(_columns)
        .collect()
    )
    return len(df)


def impl_scan_zarr_async(path: str, q: QueryBounds, base_time: datetime) -> int:
    """Query using rainbear.scan_zarr_async (no backend caching)."""
    pred = query_to_polars_pred(q, base_time)

    async def _run() -> pl.DataFrame:
        return await rainbear.scan_zarr_async(
            path,
            pl.col(_columns).filter(pred),
            max_concurrency=8
        )

    df = asyncio.run(_run())
    return len(df)


def impl_backend_async(
    backend: rainbear.ZarrBackend, q: QueryBounds, base_time: datetime
) -> int:
    """Query using ZarrBackend.scan_zarr_async (with coord caching)."""
    pred = query_to_polars_pred(q, base_time)

    async def _run() -> pl.DataFrame:
        return await backend.scan_zarr_async(
            pl.col(_columns).filter(pred),
            max_concurrency=8
        )

    df = asyncio.run(_run())
    return len(df)


def impl_backend_sync(
    backend: rainbear.ZarrBackendSync, q: QueryBounds, base_time: datetime
) -> int:
    """Query using ZarrBackendSync.scan_zarr_sync (with coord caching)."""
    pred = query_to_polars_pred(q, base_time)
    df = backend.scan_zarr_sync(
        predicate=pred,
        with_columns=_columns
    )
    return len(df)


# ---------------------------------------------------------------------------
# Multi-query runners (N different queries per benchmark iteration)
# ---------------------------------------------------------------------------


def run_n_queries_xarray_fresh(
    path: str, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries, each opening fresh xarray."""
    return [impl_xarray_fresh(path, q, base_time) for q in queries]


def run_n_queries_xarray_reused(
    ds: xr.Dataset, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries on pre-opened xarray."""
    return [impl_xarray_reused(ds, q, base_time) for q in queries]


def run_n_queries_scan_zarr(
    path: str, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries using scan_zarr."""
    return [impl_scan_zarr(path, q, base_time) for q in queries]


def run_n_queries_scan_zarr_async(
    path: str, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries using scan_zarr_async."""
    return [impl_scan_zarr_async(path, q, base_time) for q in queries]


def run_n_queries_backend_async(
    backend: rainbear.ZarrBackend, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries using ZarrBackend (async)."""
    return [impl_backend_async(backend, q, base_time) for q in queries]


def run_n_queries_backend_sync(
    backend: rainbear.ZarrBackendSync, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries using ZarrBackendSync."""
    return [impl_backend_sync(backend, q, base_time) for q in queries]


# ---------------------------------------------------------------------------
# Benchmarks: 10 novel queries per iteration
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="novel_10q")
def test_bench_novel_10q_xarray_fresh(
    benchmark, bench_dataset_path: str, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries, each opening fresh xarray."""
    results = benchmark(run_n_queries_xarray_fresh, bench_dataset_path, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_10q")
def test_bench_novel_10q_xarray_reused(
    benchmark, xarray_dataset, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries on pre-opened xarray."""
    results = benchmark(run_n_queries_xarray_reused, xarray_dataset, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_10q")
def test_bench_novel_10q_scan_zarr(
    benchmark, bench_dataset_path: str, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries using scan_zarr."""
    results = benchmark(run_n_queries_scan_zarr, bench_dataset_path, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_10q")
def test_bench_novel_10q_scan_zarr_async(
    benchmark, bench_dataset_path: str, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries using scan_zarr_async."""
    results = benchmark(run_n_queries_scan_zarr_async, bench_dataset_path, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_10q")
def test_bench_novel_10q_backend_async(
    benchmark, zarr_backend: rainbear.ZarrBackend, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries using ZarrBackend (async)."""
    results = benchmark(run_n_queries_backend_async, zarr_backend, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_10q")
def test_bench_novel_10q_backend_sync(
    benchmark, zarr_backend_sync: rainbear.ZarrBackendSync, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries using ZarrBackendSync."""
    results = benchmark(run_n_queries_backend_sync, zarr_backend_sync, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


# ---------------------------------------------------------------------------
# Benchmarks: 50 novel queries per iteration
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="novel_50q")
def test_bench_novel_50q_xarray_fresh(
    benchmark, bench_dataset_path: str, query_set_medium: list[QueryBounds], base_time: datetime
) -> None:
    """50 novel queries, each opening fresh xarray."""
    results = benchmark(run_n_queries_xarray_fresh, bench_dataset_path, query_set_medium, base_time)
    assert len(results) == 50
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_50q")
def test_bench_novel_50q_xarray_reused(
    benchmark, xarray_dataset, query_set_medium: list[QueryBounds], base_time: datetime
) -> None:
    """50 novel queries on pre-opened xarray."""
    results = benchmark(run_n_queries_xarray_reused, xarray_dataset, query_set_medium, base_time)
    assert len(results) == 50
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_50q")
def test_bench_novel_50q_scan_zarr(
    benchmark, bench_dataset_path: str, query_set_medium: list[QueryBounds], base_time: datetime
) -> None:
    """50 novel queries using scan_zarr."""
    results = benchmark(run_n_queries_scan_zarr, bench_dataset_path, query_set_medium, base_time)
    assert len(results) == 50
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_50q")
def test_bench_novel_50q_scan_zarr_async(
    benchmark, bench_dataset_path: str, query_set_medium: list[QueryBounds], base_time: datetime
) -> None:
    """50 novel queries using scan_zarr_async."""
    results = benchmark(run_n_queries_scan_zarr_async, bench_dataset_path, query_set_medium, base_time)
    assert len(results) == 50
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_50q")
def test_bench_novel_50q_backend_async(
    benchmark, zarr_backend: rainbear.ZarrBackend, query_set_medium: list[QueryBounds], base_time: datetime
) -> None:
    """50 novel queries using ZarrBackend (async)."""
    results = benchmark(run_n_queries_backend_async, zarr_backend, query_set_medium, base_time)
    assert len(results) == 50
    assert all(r > 0 for r in results)


@pytest.mark.benchmark(group="novel_50q")
def test_bench_novel_50q_backend_sync(
    benchmark, zarr_backend_sync: rainbear.ZarrBackendSync, query_set_medium: list[QueryBounds], base_time: datetime
) -> None:
    """50 novel queries using ZarrBackendSync."""
    results = benchmark(run_n_queries_backend_sync, zarr_backend_sync, query_set_medium, base_time)
    assert len(results) == 50
    assert all(r > 0 for r in results)


# ---------------------------------------------------------------------------
# Single query benchmarks (for baseline comparison)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_query(query_set_small: list[QueryBounds]) -> QueryBounds:
    """A single query for baseline benchmarks."""
    return query_set_small[0]


@pytest.mark.benchmark(group="single_novel")
def test_bench_single_xarray_fresh(
    benchmark, bench_dataset_path: str, single_query: QueryBounds, base_time: datetime
) -> None:
    """Single query opening fresh xarray."""
    result = benchmark(impl_xarray_fresh, bench_dataset_path, single_query, base_time)
    assert result > 0


@pytest.mark.benchmark(group="single_novel")
def test_bench_single_xarray_reused(
    benchmark, xarray_dataset, single_query: QueryBounds, base_time: datetime
) -> None:
    """Single query on pre-opened xarray."""
    result = benchmark(impl_xarray_reused, xarray_dataset, single_query, base_time)
    assert result > 0


@pytest.mark.benchmark(group="single_novel")
def test_bench_single_scan_zarr(
    benchmark, bench_dataset_path: str, single_query: QueryBounds, base_time: datetime
) -> None:
    """Single query using scan_zarr."""
    result = benchmark(impl_scan_zarr, bench_dataset_path, single_query, base_time)
    assert result > 0


@pytest.mark.benchmark(group="single_novel")
def test_bench_single_scan_zarr_async(
    benchmark, bench_dataset_path: str, single_query: QueryBounds, base_time: datetime
) -> None:
    """Single query using scan_zarr_async."""
    result = benchmark(impl_scan_zarr_async, bench_dataset_path, single_query, base_time)
    assert result > 0


@pytest.mark.benchmark(group="single_novel")
def test_bench_single_backend_async(
    benchmark, zarr_backend: rainbear.ZarrBackend, single_query: QueryBounds, base_time: datetime
) -> None:
    """Single query using ZarrBackend (async)."""
    result = benchmark(impl_backend_async, zarr_backend, single_query, base_time)
    assert result > 0


@pytest.mark.benchmark(group="single_novel")
def test_bench_single_backend_sync(
    benchmark, zarr_backend_sync: rainbear.ZarrBackendSync, single_query: QueryBounds, base_time: datetime
) -> None:
    """Single query using ZarrBackendSync."""
    result = benchmark(impl_backend_sync, zarr_backend_sync, single_query, base_time)
    assert result > 0


# ---------------------------------------------------------------------------
# Concurrent novel queries (async only)
# ---------------------------------------------------------------------------


def run_concurrent_queries_backend(
    backend: rainbear.ZarrBackend, queries: list[QueryBounds], base_time: datetime
) -> list[int]:
    """Run N different queries concurrently using ZarrBackend."""

    async def _run() -> list[int]:
        async def single_query(q: QueryBounds) -> int:
            pred = query_to_polars_pred(q, base_time)
            df = await backend.scan_zarr_async(
                pl.col(_columns).filter(pred),
                max_concurrency=4
            )
            return len(df)

        tasks = [single_query(q) for q in queries]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())


@pytest.mark.benchmark(group="concurrent_novel")
def test_bench_concurrent_10q_backend(
    benchmark, zarr_backend: rainbear.ZarrBackend, query_set_small: list[QueryBounds], base_time: datetime
) -> None:
    """10 novel queries run concurrently using ZarrBackend."""
    results = benchmark(run_concurrent_queries_backend, zarr_backend, query_set_small, base_time)
    assert len(results) == 10
    assert all(r > 0 for r in results)


# ---------------------------------------------------------------------------
# Remote dataset benchmarks (opt-in via env var)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def remote_dataset_path() -> str | None:
    """Get remote dataset path from env, or skip."""
    return os.environ.get("RAINBEAR_REMOTE_4D")


@pytest.fixture(scope="module")
def remote_backend(remote_dataset_path: str | None):
    """Create caching backend for remote dataset."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_4D for remote benchmarks")
    return rainbear.ZarrBackend.from_url(remote_dataset_path)


@pytest.fixture(scope="module")
def remote_xarray(remote_dataset_path: str | None):
    """Open remote xarray dataset once."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_4D for remote benchmarks")
    ds = xr.open_zarr(remote_dataset_path, chunks=None).load()
    df = ds.to_dataframe().reset_index()
    yield df
    ds.close()


@pytest.mark.benchmark(group="remote_novel_10q")
def test_bench_remote_10q_xarray_fresh(
    benchmark,
    remote_dataset_path: str | None,
    query_set_small: list[QueryBounds],
    base_time: datetime,
) -> None:
    """10 novel queries on remote, each opening fresh xarray."""
    if not remote_dataset_path:
        pytest.skip("Set RAINBEAR_REMOTE_4D for remote benchmarks")
    results = benchmark(run_n_queries_xarray_fresh, remote_dataset_path, query_set_small, base_time)
    assert len(results) == 10


@pytest.mark.benchmark(group="remote_novel_10q")
def test_bench_remote_10q_xarray_reused(
    benchmark,
    remote_xarray,
    query_set_small: list[QueryBounds],
    base_time: datetime,
) -> None:
    """10 novel queries on remote, pre-opened xarray."""
    results = benchmark(run_n_queries_xarray_reused, remote_xarray, query_set_small, base_time)
    assert len(results) == 10


@pytest.mark.benchmark(group="remote_novel_10q")
def test_bench_remote_10q_backend_async(
    benchmark,
    remote_backend,
    query_set_small: list[QueryBounds],
    base_time: datetime,
) -> None:
    """10 novel queries on remote using ZarrBackend."""
    results = benchmark(run_n_queries_backend_async, remote_backend, query_set_small, base_time)
    assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-group-by=group"])
