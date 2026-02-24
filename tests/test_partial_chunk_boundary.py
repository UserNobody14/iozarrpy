"""Tests for data integrity at partial-chunk boundaries.

When array dimensions do not evenly divide by the chunk size, the last chunk
along that axis is a *partial* chunk (e.g. a 15-wide array with chunk size 10
has a second chunk that only covers 5 elements).

These tests verify that:
1. No spurious NaN values appear at valid array positions in partial chunks.
2. Data values in partial-chunk regions match the original written data.

Bug: currently, partial chunks produce NaN values and value corruption
at certain positions, affecting both icechunk and regular zarr backends.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import rainbear
from tests.conftest import IcechunkDatasetInfo

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Icechunk backend — partial chunk tests
# ---------------------------------------------------------------------------


class TestIcechunkPartialChunkBoundary:
    """Partial-chunk integrity for the icechunk backend.

    Uses the icechunk_zlib_float64_2d dataset:
      array shape (20, 15), chunk shape (10, 10)
      → b-axis has a partial second chunk covering b=10..14
    """

    async def test_no_nan_in_valid_positions(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Every valid (a, b) position should have a non-NaN value."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        # All coordinates are within bounds, so no NaN should appear
        nan_rows = df.filter(pl.col("temperature").is_nan())
        assert nan_rows.height == 0, (
            f"Found {nan_rows.height} NaN values at valid array positions. "
            f"Affected coordinates:\n{nan_rows.select(['a', 'b']).sort(['a', 'b'])}"
        )

    async def test_partial_chunk_values_correct(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Values in the partial-chunk region (b >= 10) must match the source data."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((20, 15)).astype(np.float64) * 100.0

        # Focus on the partial chunk region: b = 10..14
        df_partial = (
            df.filter(pl.col("b") >= 10)
            .filter(pl.col("temperature").is_not_nan())
            .sort(["a", "b"])
        )

        ref_rows = []
        for a in range(20):
            for b in range(10, 15):
                ref_rows.append({"a": a, "b": b, "expected": ref[a, b]})
        ref_df = pl.DataFrame(ref_rows)

        joined = df_partial.join(ref_df, on=["a", "b"], how="inner")

        assert joined.height > 0, "No valid rows in partial chunk region"

        mismatched = joined.filter(
            (pl.col("temperature") - pl.col("expected")).abs() > 1e-10
        )
        assert mismatched.height == 0, (
            f"Found {mismatched.height} incorrect values in the partial-chunk "
            f"region (b >= 10):\n"
            f"{mismatched.select(['a', 'b', 'temperature', 'expected']).head(10)}"
        )

    async def test_total_valid_row_count(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Total non-NaN rows should equal array size (20 * 15 = 300)."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        valid_rows = df.filter(pl.col("temperature").is_not_nan()).height
        assert valid_rows == 20 * 15, (
            f"Expected 300 valid rows (20×15 array), got {valid_rows}. "
            f"NaN count: {df.height - valid_rows}"
        )

    async def test_blosc_also_affected(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """The bug is codec-independent: blosc has the same partial-chunk issue."""
        info = icechunk_datasets["icechunk_blosc_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        nan_rows = df.filter(pl.col("temperature").is_nan())
        assert nan_rows.height == 0, (
            f"Blosc backend also produces {nan_rows.height} NaN values at "
            f"partial-chunk boundaries."
        )


# ---------------------------------------------------------------------------
# Regular zarr backend — partial chunk tests
# ---------------------------------------------------------------------------


class TestZarrPartialChunkBoundary:
    """Partial-chunk integrity for the regular zarr backend.

    Uses orography_chunked_5x5:
      array shape (14, 18), chunk shape (5, 5)
      → y-axis partial at y=10..13, x-axis partial at x=15..17
    """

    async def test_no_nan_in_valid_positions(
        self,
        baseline_datasets: dict[str, str],
    ) -> None:
        """Every valid (y, x) position should have a non-NaN value."""
        path = baseline_datasets["orography_chunked_5x5"]
        backend = rainbear.ZarrBackend.from_url(path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        nan_rows = df.filter(pl.col("geopotential_height").is_nan())
        assert nan_rows.height == 0, (
            f"Found {nan_rows.height} NaN values at valid array positions. "
            f"Affected coordinates:\n{nan_rows.select(['y', 'x']).sort(['y', 'x'])}"
        )

    async def test_partial_chunk_x_boundary(
        self,
        baseline_datasets: dict[str, str],
    ) -> None:
        """Values at the x-axis partial-chunk boundary (x >= 15) must not be NaN."""
        path = baseline_datasets["orography_chunked_5x5"]
        backend = rainbear.ZarrBackend.from_url(path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        # x=15..17 is in the last partial x-chunk (chunk covers x=15..19, only 3 valid)
        partial_x = df.filter(pl.col("x") >= 15)
        nan_in_partial = partial_x.filter(
            pl.col("geopotential_height").is_nan()
        )
        assert nan_in_partial.height == 0, (
            f"Found {nan_in_partial.height} NaN values in partial x-chunk "
            f"region (x >= 15):\n{nan_in_partial.select(['y', 'x']).sort(['y', 'x'])}"
        )

    async def test_all_data_vars_affected(
        self,
        baseline_datasets: dict[str, str],
    ) -> None:
        """All float data variables should be free of spurious NaN at boundaries."""
        path = baseline_datasets["orography_chunked_5x5"]
        backend = rainbear.ZarrBackend.from_url(path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        for col in ["geopotential_height", "latitude", "longitude"]:
            nan_count = df[col].is_nan().sum()
            assert nan_count == 0, (
                f"Column '{col}' has {nan_count} spurious NaN values "
                f"at partial-chunk boundaries."
            )
