"""Regression + cloud smoke test for public ARCO ERA5 (Zarr v2 on GCS).

Set ``RAINBEAR_GCP_ARCO_ERA5_TEST=1`` to run (requires network and GCS access).

This dataset is opened with a Zarr group path that includes the store prefix
(e.g. ``/ar/.../dataset.zarr``). A past bug read data variables using only the
leaf array name, producing chunk store keys without that prefix. Zarr then
treated chunks as missing and filled with the array fill value (NaN), while
coordinate arrays still used full node paths and looked fine.

See ``sync_chunk_to_df::read_var_chunks`` (sync path); the async path already
used ``var_meta.path``.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import polars as pl
import pytest
import xarray as xr

import rainbear

ARCO_ERA5_URL = (
    "gs://gcp-public-data-arco-era5/ar/"
    "1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr"
)


@pytest.mark.skipif(
    os.environ.get("RAINBEAR_GCP_ARCO_ERA5_TEST") != "1",
    reason="Set RAINBEAR_GCP_ARCO_ERA5_TEST=1 to run this public GCS integration test.",
)
def test_arco_era5_specific_humidity_matches_xarray_point() -> None:
    t0 = datetime(2021, 4, 1, 0)
    level = 1
    latitude = 16.5
    longitude = 28.5

    ds = xr.open_zarr(ARCO_ERA5_URL)
    p = ds["specific_humidity"].sel(
        time=t0,
        level=level,
        latitude=latitude,
        longitude=longitude,
        method="nearest",
    )
    v_xr = float(p.values)
    # Predicate pushdown uses binary search on coordinate arrays. This grid stores
    # e.g. 16.5°N as 16.499999999999993 in float64, so ``== 16.5`` yields an empty
    # index range and no rows. Filter using the nearest grid values from xarray.
    time_sel = pd.Timestamp(p["time"].values.item()).to_pydatetime()
    level_sel = int(p["level"].values.item())
    lat_sel = float(p["latitude"].values.item())
    lon_sel = float(p["longitude"].values.item())

    df = (
        rainbear.scan_zarr(ARCO_ERA5_URL)
        .filter(
            (pl.col("time") == time_sel)
            & (pl.col("level") == level_sel)
            & (pl.col("latitude") == lat_sel)
            & (pl.col("longitude") == lon_sel)
        )
        .select(["specific_humidity", "latitude", "longitude", "time", "level"])
        .collect()
    )
    assert df.height == 1
    v_rb = float(df["specific_humidity"][0])
    assert v_rb == v_rb, "rainbear value is NaN (chunk read / path bug)"
    assert abs(v_rb - v_xr) <= max(1e-8, 1e-5 * abs(v_xr))
