"""Local baseline for vendor-prefixed numcodecs (Zarr v3) and Zarr v2 arrays."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

import rainbear

from tests.vendor_codec_baseline import (
    configure_rainbear_vendor_codecs,
    register_xarray_vendor_v3_codecs,
    write_forecast_like_v3_vendor_codecs,
    write_v2_zlib,
)

warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification",
)

_setup_done = False


def _ensure_vendor_codec_test_setup() -> None:
    global _setup_done
    if not _setup_done:
        register_xarray_vendor_v3_codecs()
        configure_rainbear_vendor_codecs()
        _setup_done = True


@pytest.fixture(scope="module")
def vendor_v3_forecast_zarr(tmp_path_factory: pytest.TempPathFactory) -> str:
    _ensure_vendor_codec_test_setup()
    path = tmp_path_factory.mktemp("vendor_v3") / "forecast.zarr"
    write_forecast_like_v3_vendor_codecs(path)
    return str(path)


@pytest.fixture(scope="module")
def vendor_v2_zlib_zarr(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("vendor_v2") / "v2.zarr"
    write_v2_zlib(path)
    return str(path)


def test_v3_vendor_bitround_point_matches_xarray(
    vendor_v3_forecast_zarr: str,
) -> None:
    url = vendor_v3_forecast_zarr
    ds = xr.open_zarr(url, chunks=None, consolidated=True)

    p_dew = ds["dewpoint"].sel(
        time=np.datetime64("2024-01-01T00:00:00"),
        lead_time=np.timedelta64(0, "h"),
        latitude=15.0,
        longitude=105.0,
        method="nearest",
    )
    p_air = ds["air_temperature"].sel(
        time=np.datetime64("2024-01-01T00:00:00"),
        lead_time=np.timedelta64(0, "h"),
        latitude=15.0,
        longitude=105.0,
        method="nearest",
    )

    time_sel = pd.Timestamp(p_dew["time"].values.item()).to_pydatetime()
    lead_time_sel = p_dew["lead_time"].values.item()
    lat_sel = float(p_dew["latitude"].values.item())
    lon_sel = float(p_dew["longitude"].values.item())

    df = (
        rainbear.scan_zarr(url, max_chunks_to_read=32)
        .filter(
            (pl.col("time") == time_sel)
            & (pl.col("lead_time") == lead_time_sel)
            & (pl.col("latitude") == lat_sel)
            & (pl.col("longitude") == lon_sel)
        )
        .select(
            [
                "dewpoint",
                "air_temperature",
                "latitude",
                "longitude",
                "time",
                "lead_time",
            ]
        )
        .collect()
    )
    assert df.height == 1
    v_dew_xr = float(p_dew.values)
    v_air_xr = float(p_air.values)
    v_dew_rb = float(df["dewpoint"][0])
    v_air_rb = float(df["air_temperature"][0])
    assert v_dew_rb == v_dew_rb
    assert v_air_rb == v_air_rb
    np.testing.assert_allclose(v_dew_rb, v_dew_xr, rtol=0, atol=0.2)
    np.testing.assert_allclose(v_air_rb, v_air_xr, rtol=0, atol=0.2)


def test_v3_vendor_bitround_full_grid_row_count(
    vendor_v3_forecast_zarr: str,
) -> None:
    url = vendor_v3_forecast_zarr
    ds = xr.open_zarr(url, chunks=None, consolidated=True)
    n = ds.sizes["time"] * ds.sizes["lead_time"] * ds.sizes["latitude"] * ds.sizes[
        "longitude"
    ]
    df = rainbear.scan_zarr(url, max_chunks_to_read=64).collect()
    assert df.height == n


def test_v2_zlib_roundtrip_matches_xarray(
    vendor_v2_zlib_zarr: str,
) -> None:
    url = vendor_v2_zlib_zarr
    ds = xr.open_zarr(url, chunks=None, consolidated=False)
    expected = ds["temp"].values.astype(np.float32)
    df = rainbear.scan_zarr(url, max_chunks_to_read=8).collect()
    assert df.height == len(expected)
    np.testing.assert_array_almost_equal(
        np.sort(df["temp"].to_numpy()),
        np.sort(expected),
        decimal=5,
    )
