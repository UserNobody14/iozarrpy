"""Tests for dtype preservation in IcechunkBackend with zlib/gzip codec.

Verifies that zarr float64 arrays compressed with zlib (gzip) codec are
correctly read back as Float64 polars columns, not misinterpreted as Int16
or any other type.

Also tests the CF-convention scenario where satellite data is stored as
int16 with scale_factor/add_offset, which xarray auto-decodes to float64
but rainbear reads as raw int16.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr
from zarr.codecs import GzipCodec

import rainbear
from tests.conftest import OUTPUT_DIR, IcechunkDatasetInfo
from tests.icechunk_fixtures import create_icechunk_repo_from_xarray_sync

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Reference data (must match the RNG seeds/params used in conftest fixtures)
# ---------------------------------------------------------------------------


def _make_zlib_2d_reference() -> np.ndarray:
    """Reproduce the exact float64 data written to icechunk_zlib_float64_2d."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 15)).astype(np.float64) * 100.0


def _make_zlib_3d_references() -> dict[str, np.ndarray]:
    """Reproduce the exact data written to icechunk_zlib_multidtype_3d."""
    rng = np.random.default_rng(123)
    na, nb, nc = 30, 20, 10
    return {
        "data_f64": rng.standard_normal((na, nb, nc)).astype(np.float64) * 1000.0,
        "data_f32": rng.standard_normal((na, nb, nc)).astype(np.float32) * 500.0,
        "data_i32": rng.integers(-1000, 1000, size=(na, nb, nc), dtype=np.int32),
        "data_i16": rng.integers(-500, 500, size=(na, nb, nc), dtype=np.int16),
    }


# ---------------------------------------------------------------------------
# CF-packed satellite data fixture (simulates EUMETSAT SARAH pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cf_packed_icechunk_datasets() -> dict[str, dict]:
    """Create icechunk datasets with CF-convention packing (int16 + scale_factor).

    This simulates how satellite data (e.g. EUMETSAT SARAH) is typically stored:
    - xarray writes float64 data as int16 with scale_factor/add_offset encoding
    - zarr metadata records data_type as int16
    - xarray auto-decodes back to float64 on read
    - rainbear reads the raw int16 values
    """
    icechunk_dir = OUTPUT_DIR / "icechunk"
    icechunk_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict] = {}
    na, nb = 20, 15
    rng = np.random.default_rng(77)

    # Simulate satellite calibration data (like SARAH CAL variable)
    raw_float64 = rng.standard_normal((na, nb)).astype(np.float64) * 50 + 300

    ds = xr.Dataset(
        data_vars={"CAL": (["lat", "lon"], raw_float64)},
        coords={
            "lat": np.linspace(0, 10, na),
            "lon": np.linspace(0, 20, nb),
        },
    )

    # --- CF-packed with zlib codec (matches EUMETSAT SARAH pattern) ---
    path = create_icechunk_repo_from_xarray_sync(
        ds,
        icechunk_dir / "cf_packed_zlib.icechunk",
        encoding={
            "CAL": {
                "dtype": "int16",
                "scale_factor": 0.01,
                "add_offset": 300.0,
                "_FillValue": np.int16(-32767),
                "chunks": (10, 10),
                "compressors": [GzipCodec(level=5)],
            }
        },
    )
    datasets["cf_packed_zlib"] = {
        "path": path,
        "original_float64": raw_float64,
        "scale_factor": 0.01,
        "add_offset": 300.0,
    }

    # --- CF-packed with default codec (control) ---
    path = create_icechunk_repo_from_xarray_sync(
        ds,
        icechunk_dir / "cf_packed_default.icechunk",
        encoding={
            "CAL": {
                "dtype": "int16",
                "scale_factor": 0.01,
                "add_offset": 300.0,
                "_FillValue": np.int16(-32767),
                "chunks": (10, 10),
            }
        },
    )
    datasets["cf_packed_default"] = {
        "path": path,
        "original_float64": raw_float64,
        "scale_factor": 0.01,
        "add_offset": 300.0,
    }

    # --- Pure float64 with zlib codec (no CF packing) ---
    path = create_icechunk_repo_from_xarray_sync(
        ds,
        icechunk_dir / "pure_float64_zlib.icechunk",
        encoding={
            "CAL": {
                "chunks": (10, 10),
                "compressors": [GzipCodec(level=5)],
            }
        },
    )
    datasets["pure_float64_zlib"] = {
        "path": path,
        "original_float64": raw_float64,
    }

    return datasets


# ---------------------------------------------------------------------------
# Schema dtype tests
# ---------------------------------------------------------------------------


class TestZlibSchemaDtype:
    """Verify that schema reports correct dtypes for zlib-compressed arrays."""

    async def test_schema_float64_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Schema should report Float64 for a float64 array using zlib codec."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)
        schema = backend.schema()

        assert "temperature" in schema
        assert schema["temperature"] == pl.Float64, (
            f"Expected Float64 for temperature, got {schema['temperature']}"
        )

    async def test_schema_multidtype_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Schema should report correct dtypes for multiple zarr dtypes with zlib."""
        info = icechunk_datasets["icechunk_zlib_multidtype_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)
        schema = backend.schema()

        expected = {
            "data_f64": pl.Float64,
            "data_f32": pl.Float32,
            "data_i32": pl.Int32,
            "data_i16": pl.Int16,
        }
        for var, expected_dtype in expected.items():
            assert var in schema, f"Missing variable: {var}"
            assert schema[var] == expected_dtype, (
                f"Expected {expected_dtype} for {var}, got {schema[var]}"
            )


# ---------------------------------------------------------------------------
# Scan dtype tests — the core bug verification
# ---------------------------------------------------------------------------


class TestZlibScanDtype:
    """Verify that scanned DataFrames have correct column dtypes with zlib codec."""

    async def test_scan_float64_dtype_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Float64 zarr array with zlib should produce Float64 polars column."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df.height > 0, "Scan returned empty DataFrame"
        assert df["temperature"].dtype == pl.Float64, (
            f"Expected Float64 for temperature column, got {df['temperature'].dtype}. "
            f"This indicates the zlib codec is causing dtype misinterpretation."
        )

    async def test_scan_float64_dtype_not_i16_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Explicitly verify that float64 data with zlib is NOT read as Int16."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df["temperature"].dtype != pl.Int16, (
            "Float64 zarr data with zlib codec was misinterpreted as Int16! "
            "This is the known zlib dtype bug."
        )

    async def test_scan_all_dtypes_preserved_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """All zarr dtypes should be preserved when using zlib compression."""
        info = icechunk_datasets["icechunk_zlib_multidtype_3d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df.height > 0, "Scan returned empty DataFrame"

        expected = {
            "data_f64": pl.Float64,
            "data_f32": pl.Float32,
            "data_i32": pl.Int32,
            "data_i16": pl.Int16,
        }
        for var, expected_dtype in expected.items():
            assert var in df.columns, f"Missing column: {var}"
            actual_dtype = df[var].dtype
            assert actual_dtype == expected_dtype, (
                f"Expected {expected_dtype} for {var}, got {actual_dtype}. "
                f"Zlib codec is causing dtype corruption."
            )

    async def test_scan_float64_with_predicate_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Float64 dtype should be preserved even with predicate pushdown and zlib."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        pred = pl.col("a") < 10
        df = await backend.scan_zarr_async(
            pl.col(["a", "b", "temperature"]).filter(pred)
        )

        df_filtered = df.filter(pred)
        assert df_filtered.height > 0
        assert df_filtered["temperature"].dtype == pl.Float64, (
            f"Expected Float64 after predicate pushdown with zlib, "
            f"got {df_filtered['temperature'].dtype}"
        )


# ---------------------------------------------------------------------------
# Value correctness tests
# ---------------------------------------------------------------------------


class TestZlibValueCorrectness:
    """Verify that scanned values are correct, not garbled by dtype misinterpretation."""

    async def test_float64_values_correct_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Float64 values should be numerically correct after zlib decode."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))
        ref = _make_zlib_2d_reference()

        # Restrict to the first complete chunk to avoid partial-chunk boundary issues
        df_chunk0 = df.filter(
            (pl.col("a") < 10) & (pl.col("b") < 10)
        ).sort(["a", "b"])

        ref_rows = []
        for a in range(10):
            for b in range(10):
                ref_rows.append({"a": a, "b": b, "expected": ref[a, b]})
        ref_df = pl.DataFrame(ref_rows)

        joined = df_chunk0.join(ref_df, on=["a", "b"], how="inner")

        assert joined.height == 100, (
            f"Expected 100 rows in first chunk, got {joined.height}"
        )

        np.testing.assert_allclose(
            joined["temperature"].to_numpy(),
            joined["expected"].to_numpy(),
            rtol=1e-10,
            err_msg=(
                "Float64 values from zlib-compressed icechunk data do not match "
                "the original values. This suggests the bytes are being "
                "misinterpreted (e.g. as Int16 instead of Float64)."
            ),
        )

    async def test_float64_values_have_fractional_parts(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Float64 data should contain fractional values (not truncated to integers)."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        temp = df["temperature"].drop_nans()
        fractional_count = (temp - temp.cast(pl.Int64).cast(pl.Float64)).abs().gt(0.0).sum()
        assert fractional_count > 0, (
            "All temperature values are exact integers, suggesting float64 data "
            "was truncated or misinterpreted as an integer type."
        )

    async def test_float64_value_range_reasonable(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Float64 values should be in a reasonable range, not i16 garbage."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        temp = df["temperature"].drop_nans()
        ref = _make_zlib_2d_reference()

        assert temp.min() >= ref.min() - 1.0, (
            f"Temperature min {temp.min()} is way off from expected {ref.min()}"
        )
        assert temp.max() <= ref.max() + 1.0, (
            f"Temperature max {temp.max()} is way off from expected {ref.max()}"
        )


# ---------------------------------------------------------------------------
# Codec comparison tests (zlib vs blosc on identical data)
# ---------------------------------------------------------------------------


class TestZlibVsBloscEquivalence:
    """Compare zlib and blosc codecs on the same data to isolate the codec as cause."""

    async def test_zlib_vs_blosc_same_dtype(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Same float64 data with zlib vs blosc should produce same polars dtype."""
        info_zlib = icechunk_datasets["icechunk_zlib_float64_2d"]
        info_blosc = icechunk_datasets["icechunk_blosc_float64_2d"]

        backend_zlib = await rainbear.IcechunkBackend.from_filesystem(info_zlib.path)
        backend_blosc = await rainbear.IcechunkBackend.from_filesystem(info_blosc.path)

        df_zlib = await backend_zlib.scan_zarr_async(pl.all().filter(pl.lit(True)))
        df_blosc = await backend_blosc.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df_zlib["temperature"].dtype == df_blosc["temperature"].dtype, (
            f"Dtype mismatch between codecs: "
            f"zlib={df_zlib['temperature'].dtype}, blosc={df_blosc['temperature'].dtype}. "
            f"Both should be Float64."
        )

    async def test_zlib_vs_blosc_same_values(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Same float64 data with zlib vs blosc should produce identical values."""
        info_zlib = icechunk_datasets["icechunk_zlib_float64_2d"]
        info_blosc = icechunk_datasets["icechunk_blosc_float64_2d"]

        backend_zlib = await rainbear.IcechunkBackend.from_filesystem(info_zlib.path)
        backend_blosc = await rainbear.IcechunkBackend.from_filesystem(info_blosc.path)

        df_zlib = await backend_zlib.scan_zarr_async(pl.all().filter(pl.lit(True)))
        df_blosc = await backend_blosc.scan_zarr_async(pl.all().filter(pl.lit(True)))

        df_zlib = df_zlib.sort(["a", "b"])
        df_blosc = df_blosc.sort(["a", "b"])

        assert df_zlib.height == df_blosc.height, (
            f"Row count mismatch: zlib={df_zlib.height}, blosc={df_blosc.height}"
        )

        zlib_vals = df_zlib["temperature"].to_numpy()
        blosc_vals = df_blosc["temperature"].to_numpy()

        # Handle NaN comparison
        mask = ~(np.isnan(zlib_vals) | np.isnan(blosc_vals))
        np.testing.assert_allclose(
            zlib_vals[mask],
            blosc_vals[mask],
            rtol=1e-10,
            err_msg=(
                "Values differ between zlib and blosc codecs on identical data. "
                "This confirms the zlib codec is corrupting data."
            ),
        )

    async def test_zlib_vs_blosc_schema_match(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Schema should be identical regardless of codec (zlib vs blosc)."""
        info_zlib = icechunk_datasets["icechunk_zlib_float64_2d"]
        info_blosc = icechunk_datasets["icechunk_blosc_float64_2d"]

        backend_zlib = await rainbear.IcechunkBackend.from_filesystem(info_zlib.path)
        backend_blosc = await rainbear.IcechunkBackend.from_filesystem(info_blosc.path)

        schema_zlib = backend_zlib.schema()
        schema_blosc = backend_blosc.schema()

        assert schema_zlib == schema_blosc, (
            f"Schema mismatch: zlib={schema_zlib}, blosc={schema_blosc}"
        )


# ---------------------------------------------------------------------------
# CF-packed data tests (satellite data pattern: int16 + scale_factor/add_offset)
#
# This is the most likely scenario causing the user's issue:
# - EUMETSAT SARAH data stores CAL as int16 with scale_factor/add_offset
# - xarray auto-decodes to float64 (user sees "float64 zarr values")
# - rainbear reads raw int16 (user sees "i16 polars columns")
# ---------------------------------------------------------------------------


class TestCFPackedDtype:
    """Test that CF-packed data (int16 + scale_factor/add_offset) is handled.

    When satellite data like EUMETSAT SARAH stores float64 data as packed int16
    with CF-convention scale_factor/add_offset attributes:
    - xarray shows float64 (auto-decoded)
    - zarr metadata records int16 as the data_type
    - rainbear should read the raw zarr dtype (int16)

    These tests document the expected behavior and will help determine whether
    the issue is a codec bug or a CF-decoding feature gap.
    """

    async def test_cf_packed_zarr_dtype_is_int16(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """CF-packed zarr array has int16 dtype in zarr metadata."""
        info = cf_packed_icechunk_datasets["cf_packed_zlib"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info["path"])
        schema = backend.schema()

        # The zarr metadata says int16 (CF packing), so schema should reflect that
        assert "CAL" in schema
        assert schema["CAL"] == pl.Int16, (
            f"Expected Int16 for CF-packed CAL (raw zarr dtype), got {schema['CAL']}. "
            f"The zarr data_type is int16 even though xarray shows float64."
        )

    async def test_cf_packed_scan_returns_int16(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """Scanning CF-packed data returns int16 (the raw zarr dtype)."""
        info = cf_packed_icechunk_datasets["cf_packed_zlib"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info["path"])

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df.height > 0
        assert df["CAL"].dtype == pl.Int16, (
            f"Expected Int16 for CF-packed CAL, got {df['CAL'].dtype}. "
            f"rainbear reads raw zarr data; int16 is correct for CF-packed data."
        )

    async def test_cf_packed_values_can_be_decoded_to_float64(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """CF-packed int16 values can be manually decoded to float64 using attributes."""
        info = cf_packed_icechunk_datasets["cf_packed_zlib"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info["path"])

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        scale = info["scale_factor"]
        offset = info["add_offset"]
        original = info["original_float64"]

        # Restrict to first complete chunk to avoid partial-chunk boundary issues.
        # lat has 20 values with chunk size 10, lon has 15 with chunk size 10.
        lat_vals = np.linspace(0, 10, 20)
        lon_vals = np.linspace(0, 20, 15)
        lat_max_chunk0 = lat_vals[9]
        lon_max_chunk0 = lon_vals[9]

        df_chunk0 = df.filter(
            (pl.col("lat") <= lat_max_chunk0) & (pl.col("lon") <= lon_max_chunk0)
            & (pl.col("CAL") != -32767)
        ).sort(["lat", "lon"])

        assert df_chunk0.height > 0, "No rows in first chunk"

        decoded = (df_chunk0["CAL"].cast(pl.Float64) * scale + offset).to_numpy()

        ref_rows = []
        for i in range(10):
            for j in range(10):
                ref_rows.append(original[i, j])

        np.testing.assert_allclose(
            decoded,
            np.array(ref_rows),
            atol=scale,
            err_msg="CF-decoded values don't match original float64 data",
        )

    async def test_cf_packed_xarray_shows_float64_but_zarr_is_int16(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """Demonstrate the xarray vs rainbear dtype discrepancy for CF-packed data.

        This test documents the root cause of confusion: xarray auto-decodes
        CF-packed int16 to float64, while rainbear correctly reads the raw
        zarr int16 values.
        """
        info = cf_packed_icechunk_datasets["cf_packed_zlib"]
        path = info["path"]

        # xarray reads float64 (auto-decoded)
        from icechunk import Repository, local_filesystem_storage
        storage = local_filesystem_storage(path)
        repo = Repository.open(storage)
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, zarr_version=3, consolidated=False)
        xarray_dtype = ds["CAL"].dtype

        # rainbear reads int16 (raw zarr dtype)
        backend = await rainbear.IcechunkBackend.from_filesystem(path)
        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))
        rainbear_dtype = df["CAL"].dtype

        # This documents the discrepancy:
        # xarray auto-decodes CF scale_factor/add_offset → float64
        # rainbear reads raw zarr data → int16
        assert xarray_dtype == np.float64, (
            f"xarray should show float64 (CF-decoded), got {xarray_dtype}"
        )
        assert rainbear_dtype == pl.Int16, (
            f"rainbear should show Int16 (raw zarr dtype), got {rainbear_dtype}"
        )

    async def test_pure_float64_zlib_produces_float64(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """Pure float64 zarr data with zlib should produce Float64 in rainbear."""
        info = cf_packed_icechunk_datasets["pure_float64_zlib"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info["path"])

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df["CAL"].dtype == pl.Float64, (
            f"Pure float64 data with zlib should produce Float64, got {df['CAL'].dtype}"
        )

    async def test_pure_float64_zlib_values_correct(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """Pure float64 values with zlib should be numerically correct."""
        info = cf_packed_icechunk_datasets["pure_float64_zlib"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info["path"])

        df = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))
        original = info["original_float64"]

        # Restrict to first complete chunk to avoid partial-chunk boundary issues
        lat_vals = np.linspace(0, 10, 20)
        lon_vals = np.linspace(0, 20, 15)
        lat_max_chunk0 = lat_vals[9]
        lon_max_chunk0 = lon_vals[9]

        df_chunk0 = df.filter(
            (pl.col("lat") <= lat_max_chunk0) & (pl.col("lon") <= lon_max_chunk0)
            & pl.col("CAL").is_not_nan()
        ).sort(["lat", "lon"])

        assert df_chunk0.height > 0, "No rows in first chunk"

        ref_rows = []
        for i in range(10):
            for j in range(10):
                ref_rows.append(original[i, j])

        np.testing.assert_allclose(
            df_chunk0["CAL"].to_numpy(),
            np.array(ref_rows),
            rtol=1e-10,
            err_msg="Pure float64 values with zlib don't match original data",
        )

    async def test_cf_packed_zlib_vs_default_codec_same_dtype(
        self,
        cf_packed_icechunk_datasets: dict[str, dict],
    ) -> None:
        """CF-packed data should have same dtype regardless of codec (zlib vs default)."""
        info_zlib = cf_packed_icechunk_datasets["cf_packed_zlib"]
        info_default = cf_packed_icechunk_datasets["cf_packed_default"]

        backend_zlib = await rainbear.IcechunkBackend.from_filesystem(info_zlib["path"])
        backend_default = await rainbear.IcechunkBackend.from_filesystem(info_default["path"])

        df_zlib = await backend_zlib.scan_zarr_async(pl.all().filter(pl.lit(True)))
        df_default = await backend_default.scan_zarr_async(pl.all().filter(pl.lit(True)))

        assert df_zlib["CAL"].dtype == df_default["CAL"].dtype, (
            f"Dtype mismatch: zlib={df_zlib['CAL'].dtype}, default={df_default['CAL'].dtype}"
        )


# ---------------------------------------------------------------------------
# Streaming iterator dtype tests
# ---------------------------------------------------------------------------


class TestZlibStreamingDtype:
    """Verify dtype preservation in the streaming iterator path with zlib codec."""

    async def test_streaming_float64_dtype_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Streaming iterator should preserve Float64 with zlib codec."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        batches = list(backend.scan_zarr_streaming_sync(batch_size=100))
        assert len(batches) > 0, "Streaming scan returned no batches"

        for i, batch in enumerate(batches):
            assert batch["temperature"].dtype == pl.Float64, (
                f"Batch {i}: Expected Float64, got {batch['temperature'].dtype}. "
                f"Streaming zlib decode is corrupting dtypes."
            )

    async def test_streaming_float64_not_i16_with_zlib(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Streaming iterator with zlib should NOT produce Int16 for float64 data."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        batches = list(backend.scan_zarr_streaming_sync(batch_size=100))
        assert len(batches) > 0

        for i, batch in enumerate(batches):
            assert batch["temperature"].dtype != pl.Int16, (
                f"Batch {i}: Float64 data was misinterpreted as Int16 in streaming mode!"
            )

    async def test_streaming_values_match_async_scan(
        self,
        icechunk_datasets: dict[str, IcechunkDatasetInfo],
    ) -> None:
        """Streaming and async scan should produce identical results with zlib."""
        info = icechunk_datasets["icechunk_zlib_float64_2d"]
        backend = await rainbear.IcechunkBackend.from_filesystem(info.path)

        df_async = await backend.scan_zarr_async(pl.all().filter(pl.lit(True)))

        batches = list(backend.scan_zarr_streaming_sync(batch_size=1000))
        df_streaming = pl.concat(batches)

        df_async = df_async.sort(["a", "b"])
        df_streaming = df_streaming.sort(["a", "b"])

        assert df_async.height == df_streaming.height

        assert df_async["temperature"].dtype == df_streaming["temperature"].dtype, (
            f"Dtype mismatch: async={df_async['temperature'].dtype}, "
            f"streaming={df_streaming['temperature'].dtype}"
        )

        actual_async = df_async["temperature"].to_numpy()
        actual_stream = df_streaming["temperature"].to_numpy()
        mask = ~(np.isnan(actual_async) | np.isnan(actual_stream))
        np.testing.assert_allclose(
            actual_async[mask],
            actual_stream[mask],
            rtol=1e-10,
        )
