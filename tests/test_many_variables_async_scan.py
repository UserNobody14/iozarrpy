"""Regression: many variables on one grid (ERA5-style) with async scan.

Previously, async (and sync) chunk reads cloned every decoded chunk after the
backend already returned an ``Arc``, doubling memory and CPU for each variable.
This test builds a small 4D zarr with 13 variables and matching chunking, then
checks ``scan_zarr_async`` against ``scan_zarr``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

import rainbear

# Same variable set as a typical multi-field weather grid (user-reported case).
_MANY_VAR_NAMES: tuple[str, ...] = (
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "100m_wind_speed",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "prob_of_precipitation_1hr",
    "surface_solar_radiation_downwards",
    "total_cloud_cover",
    "total_precipitation_1hr",
)


def _normalize(df: pl.DataFrame) -> list[dict[str, object]]:
    cols = sorted(df.columns)
    df = df.select(cols)
    if df.height == 0:
        return []
    df = df.sort(cols)
    return df.to_dicts()


@pytest.fixture
def many_var_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """4D grid with 1D coords and chunk layout similar to operational ERA5 subsets."""
    path = tmp_path_factory.mktemp("many_var") / "era5_like.zarr"
    nt, nl, nlat, nlon = 4, 3, 72, 144
    chunks = (1, 1, 18, 36)
    rng = np.random.default_rng(0)

    data_vars = {
        name: (
            ["time", "lead_time", "latitude", "longitude"],
            rng.standard_normal((nt, nl, nlat, nlon), dtype=np.float32),
        )
        for name in _MANY_VAR_NAMES
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.arange(nt, dtype=np.int64),
            "lead_time": np.arange(nl, dtype=np.int64),
            "latitude": np.linspace(-90.0, 90.0, nlat, dtype=np.float32),
            "longitude": np.linspace(0.0, 359.0, nlon, dtype=np.float32),
        },
    )
    encoding = {name: {"chunks": chunks} for name in _MANY_VAR_NAMES}
    ds.to_zarr(path, zarr_format=3, encoding=encoding)
    return str(path)


@pytest.mark.asyncio
async def test_scan_zarr_async_many_variables_matches_sync(
    many_var_zarr_path: str,
) -> None:
    """Restrict to one chunk in index space; compare lazy sync vs async."""
    pred = (
        (pl.col("time") == 0)
        & (pl.col("lead_time") == 0)
        & (pl.col("latitude") >= -90.0)
        & (pl.col("latitude") <= -72.0)
        & (pl.col("longitude") >= 0.0)
        & (pl.col("longitude") <= 35.0)
    )
    cols = [
        "time",
        "lead_time",
        "latitude",
        "longitude",
        *_MANY_VAR_NAMES,
    ]

    df_sync = (
        rainbear.scan_zarr(many_var_zarr_path)
        .filter(pred)
        .select(cols)
        .collect()
    )
    df_async = (
        await rainbear.scan_zarr_async(
            many_var_zarr_path,
            pl.col(cols).filter(pred),
            max_concurrency=16,
        )
    ).filter(pred)

    assert _normalize(df_async) == _normalize(df_sync)
