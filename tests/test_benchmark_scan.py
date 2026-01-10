"""Benchmarks for scanning Zarr -> Polars.

Compares:
- manual xarray -> pandas -> polars
- rainbear.scan_zarr (LazyFrame IO source)
- rainbear.scan_zarr_async (awaitable, parallel async reads)
"""

from __future__ import annotations

import asyncio

import polars as pl
import pytest
import xarray as xr

import rainbear


@pytest.fixture(scope="session")
def bench_dataset_path(baseline_datasets: dict[str, str]) -> str:
    # Use a baseline dataset so xarray can open it (has dimension metadata),
    # and keep the benchmark bounded by selecting a small subset.
    return baseline_datasets["orography_chunked_10x10"]


@pytest.fixture(autouse=True)
def _bench_enable_pushdown(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest disables pushdown for correctness tests; for benchmarks we want the
    # default/realistic behavior (pushdown enabled when possible).
    monkeypatch.setenv("RAINBEAR_PREDICATE_PUSHDOWN", "1")


def _pred_orog_subset() -> pl.Expr:
    return (
        (pl.col("y") >= 3)
        & (pl.col("y") <= 10)
        & (pl.col("x") >= 4)
        & (pl.col("x") <= 12)
    )


def _bench_xarray_to_polars(path: str) -> pl.DataFrame:
    # Force real IO every iteration (avoid reusing an already-open dataset).
    ds = xr.open_zarr(path, consolidated=False)
    try:
        pdf = ds[["geopotential_height"]].to_dataframe().reset_index()
    finally:
        ds.close()
    df = pl.from_pandas(pdf)
    return df.filter(_pred_orog_subset()).select(["y", "x", "geopotential_height"])


def _bench_rainbear_scan_zarr(path: str) -> pl.DataFrame:
    pred = _pred_orog_subset()
    return (
        rainbear.scan_zarr(path, variables=["geopotential_height"])
        .filter(pred)
        .select(["y", "x", "geopotential_height"])
        .collect()
    )


def _bench_rainbear_scan_zarr_async(path: str) -> pl.DataFrame:
    pred = _pred_orog_subset()

    async def _run() -> pl.DataFrame:
        df = await rainbear.scan_zarr_async(
            path,
            pred,
            variables=["geopotential_height"],
            max_concurrency=8,
            with_columns=["y", "x", "geopotential_height"],
        )
        # scan_zarr_async uses the predicate for planning, but row filtering is done here.
        return df.filter(pred).select(["y", "x", "geopotential_height"])

    return asyncio.run(_run())


@pytest.mark.benchmark(group="orography_chunked_10x10_subset")
@pytest.mark.parametrize(
    "impl",
    [
        pytest.param("xarray_to_polars", id="xarray->polars"),
        pytest.param("rainbear_scan_zarr", id="scan_zarr"),
        pytest.param("rainbear_scan_zarr_async", id="scan_zarr_async"),
    ],
)
def test_bench_orography_chunked_10x10_subset(
    benchmark, bench_dataset_path: str, impl: str
) -> None:
    # Keep results alive so benchmark can't optimize away work.
    fn = {
        "xarray_to_polars": _bench_xarray_to_polars,
        "rainbear_scan_zarr": _bench_rainbear_scan_zarr,
        "rainbear_scan_zarr_async": _bench_rainbear_scan_zarr_async,
    }[impl]

    out = benchmark(fn, bench_dataset_path)
    assert out.columns == ["y", "x", "geopotential_height"]

