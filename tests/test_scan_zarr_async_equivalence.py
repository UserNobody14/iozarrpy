"""Equivalence tests for scan_zarr_async vs scan_zarr.

We validate correctness by comparing the final *filtered* result sets.

Note: scan_zarr_async currently uses the predicate for chunk planning (pruning),
but does not apply the predicate as a row filter internally. Therefore the tests
explicitly apply the predicate after awaiting.
"""

from __future__ import annotations

import polars as pl
import pytest

import rainbear


def _normalize(df: pl.DataFrame) -> list[dict[str, object]]:
    # Sort both columns and rows for stable comparisons.
    cols = sorted(df.columns)
    df = df.select(cols)
    if df.height == 0:
        return []
    df = df.sort(cols)
    return df.to_dicts()


@pytest.mark.asyncio
async def test_scan_zarr_async_matches_sync_orography_chunked_subset(
    baseline_datasets: dict[str, str],
) -> None:
    zarr_url = baseline_datasets["orography_chunked_10x10"]
    pred = (pl.col("y") >= 3) & (pl.col("y") <= 10) & (pl.col("x") >= 4) & (pl.col("x") <= 12)
    cols = ["y", "x", "geopotential_height"]

    df_sync = (
        rainbear.scan_zarr(zarr_url)
        .filter(pred)
        .select(cols)
        .collect()
    )
    df_async = (
        (
            await rainbear.scan_zarr_async(
                zarr_url,
                pred,
                variables=["geopotential_height"],
                with_columns=cols,
                max_concurrency=8,
            )
        ).filter(pred)
    )

    assert _normalize(df_async) == _normalize(df_sync)


@pytest.mark.asyncio
async def test_scan_zarr_async_matches_sync_orography_sharded_subset(
    baseline_datasets: dict[str, str],
) -> None:
    # Use a small 2D dataset so CI doesn't end up scanning millions of rows.
    zarr_url = baseline_datasets["orography_sharded_small"]

    pred = (pl.col("y") >= 3) & (pl.col("y") <= 10) & (pl.col("x") >= 4) & (pl.col("x") <= 12)
    cols = ["y", "x", "geopotential_height"]

    df_sync = (
        rainbear.scan_zarr(zarr_url)
        .filter(pred)
        .select(cols)
        .collect()
    )

    df_async = (
        (
            await rainbear.scan_zarr_async(
                zarr_url,
                pred,
                variables=["geopotential_height"],
                with_columns=cols,
                max_concurrency=8,
            )
        ).filter(pred)
    )

    assert _normalize(df_async) == _normalize(df_sync)


