"""Streaming scan with (time, lead_time, point) layout — replica of auxiliary coord 'point' datasets.

Regression: Polars IO projection pushdown must not drop columns only referenced in `.filter()`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

import rainbear

_REPO_NZ_ZARR = Path(__file__).resolve().parents[1] / "sandbox" / "nz.zarr"


def _write_nz_style_point_dataset(path: str) -> xr.Dataset:
    """Minimal dataset matching sandbox/nz.zarr structure (shared ``point`` dim, consolidated zarr v3)."""
    n_t, n_lt, n_pt = 2, 3, 4
    ref = "hours since 2024-01-01"
    # Two valid time indices; rainbear decodes to datetime[ns]
    time = xr.DataArray(
        np.array([0, 24], dtype=np.int64),
        dims=("time",),
        attrs={"units": ref, "calendar": "proleptic_gregorian"},
    )
    lead_time = xr.DataArray(
        np.arange(n_lt, dtype=np.int64),
        dims=("lead_time",),
        attrs={"units": "hours"},
    )
    lat = xr.DataArray(
        np.linspace(-45.0, -33.0, n_pt, dtype=np.float64),
        dims=("point",),
    )
    lon = xr.DataArray(
        np.linspace(168.0, 178.0, n_pt, dtype=np.float64),
        dims=("point",),
    )
    # Avoid xarray fixed_length_utf32 strings (unsupported in zarrs); use int ids.
    station_id = xr.DataArray(
        np.arange(n_pt, dtype=np.int64),
        dims=("point",),
        attrs={"coordinates": "latitude longitude"},
    )
    vals = np.arange(n_t * n_lt * n_pt, dtype=np.float32).reshape(
        n_t,
        n_lt,
        n_pt,
    )
    ds = xr.Dataset(
        {
            "rime_ice_probability_1hr": (
                ("time", "lead_time", "point"),
                vals,
            ),
            "latitude": lat,
            "longitude": lon,
            "station_id": station_id,
            "time": time,
            "lead_time": lead_time,
        },
    )
    ds["rime_ice_probability_1hr"].attrs["coordinates"] = "latitude longitude"
    ds.to_zarr(path, zarr_format=3, consolidated=True)
    return ds


@pytest.fixture(scope="module")
def nz_style_point_zarr(tmp_path_factory: pytest.TempPathFactory) -> str:
    out = tmp_path_factory.mktemp("nz_point") / "replica.zarr"
    _write_nz_style_point_dataset(str(out))
    return str(out)


def test_streaming_scan_filter_then_select_point_dims(
    nz_style_point_zarr: str,
) -> None:
    n_pt = 4
    lf = (
        rainbear.scan_zarr(nz_style_point_zarr)
        .filter(
            (pl.col("time") == pl.col("time").min())
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
        .select(["rime_ice_probability_1hr", "time", "station_id"])
    )
    df = lf.collect()

    assert df.height == n_pt
    assert df.columns == [
        "rime_ice_probability_1hr",
        "time",
        "station_id",
    ]
    assert df["station_id"].dtype == pl.Int64


def test_general_realistic(nz_style_point_zarr: str) -> None:
    """Filter + multi-column select on streaming IO; auxiliary grid is not iterated alone."""
    hqb = rainbear.ZarrBackendSync.from_url(nz_style_point_zarr)
    lf = (
        rainbear.scan_zarr(hqb)
        .filter(
            (pl.col("time") == pl.col("time").min())
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
        .select(["time", "point", "station_id", "rime_ice_probability_1hr"])
    )
    df = lf.collect()
    assert df.height == 4
    assert df.columns == [
        "time",
        "point",
        "station_id",
        "rime_ice_probability_1hr",
    ]
    assert df["station_id"].dtype == pl.Int64
    assert df["station_id"].null_count() == 0
    assert df["station_id"].to_list() == [0, 1, 2, 3]
    # C-order (t, lead_time, point): t=0, lead=1 -> base 4 + point
    assert df["rime_ice_probability_1hr"].to_list() == [4.0, 5.0, 6.0, 7.0]


def test_alt_realistic(nz_style_point_zarr: str) -> None:
    hqb = rainbear.ZarrBackendSync.from_url(nz_style_point_zarr)
    lf = (
        rainbear.scan_zarr(hqb)
        .filter(
            (pl.col("time") == datetime(2024, 1, 1, 0))
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
        .select(
            [
                "time",
                "station_id",
                "rime_ice_probability_1hr",
                "lead_time",
                "latitude",
                "longitude",
            ]
        )
    )
    df = lf.collect()
    assert df.height == 4
    assert df.columns == [
        "time",
        "station_id",
        "rime_ice_probability_1hr",
        "lead_time",
        "latitude",
        "longitude",
    ]
    assert df["station_id"].dtype == pl.Int64
    assert df["station_id"].null_count() == 0
    assert df["station_id"].to_list() == [0, 1, 2, 3]
    assert df["rime_ice_probability_1hr"].to_list() == [4.0, 5.0, 6.0, 7.0]
    n_pt = 4
    assert df["latitude"].to_list() == pytest.approx(
        np.linspace(-45.0, -33.0, n_pt).tolist()
    )
    assert df["longitude"].to_list() == pytest.approx(
        np.linspace(168.0, 178.0, n_pt).tolist()
    )


def test_alt_realistic_pointed(nz_style_point_zarr: str) -> None:
    hqb = rainbear.ZarrBackendSync.from_url(nz_style_point_zarr)
    lf = (
        rainbear.scan_zarr(hqb)
        .filter(
            (pl.col("time") == datetime(2024, 1, 1, 0))
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
        .select(
            [
                "time",
                "station_id",
                "point",
                "rime_ice_probability_1hr",
                "lead_time",
                "latitude",
                "longitude",
            ]
        )
    )
    df = lf.collect()
    assert df.height == 4
    assert df.columns == [
        "time",
        "station_id",
        "point",
        "rime_ice_probability_1hr",
        "lead_time",
        "latitude",
        "longitude",
    ]
    assert df["station_id"].dtype == pl.Int64
    assert df["station_id"].null_count() == 0
    assert df["station_id"].to_list() == [0, 1, 2, 3]
    assert df["rime_ice_probability_1hr"].to_list() == [4.0, 5.0, 6.0, 7.0]
    n_pt = 4
    assert df["latitude"].to_list() == pytest.approx(
        np.linspace(-45.0, -33.0, n_pt).tolist()
    )
    assert df["longitude"].to_list() == pytest.approx(
        np.linspace(168.0, 178.0, n_pt).tolist()
    )



def test_alt_realistic_chunks_debug(nz_style_point_zarr: str) -> None:
    hqb = rainbear.ZarrBackendSync.from_url(nz_style_point_zarr)
    scdb = hqb.selected_chunks_debug(
        pl.col(
            [
                "time",
                "station_id",
                "rime_ice_probability_1hr",
                "lead_time",
                "latitude",
                "longitude",
            ]
        ).filter(
            (pl.col("time") == datetime(2024, 1, 1, 0))
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
    )
    assert scdb["grids"] == [
        {
            "dims": ["lead_time"],
            "variables": ["lead_time"],
            "chunks": [{"indices": [0], "origin": [0], "shape": [3], "shards": []}],
        },
        {
            "dims": ["point"],
            "variables": ["latitude", "longitude", "station_id"],
            "chunks": [{"indices": [0], "origin": [0], "shape": [4], "shards": []}],
        },
        {
            "dims": ["time"],
            "variables": ["time"],
            "chunks": [{"indices": [0], "origin": [0], "shape": [2], "shards": []}],
        },
        {
            "dims": ["time", "lead_time", "point"],
            "variables": ["rime_ice_probability_1hr"],
            "chunks": [{"indices": [0, 0, 0], "origin": [0, 0, 0], "shape": [2, 3, 4], "shards": []}],
        },
    ]
    assert scdb["coord_reads"] == 0



def test_streaming_matches_sync_scan_predicate(
    nz_style_point_zarr: str,
) -> None:
    hqb = rainbear.ZarrBackendSync.from_url(nz_style_point_zarr)
    prd = pl.col("lead_time") == timedelta(hours=0)
    df_sync = hqb.scan_zarr_sync(predicate=prd)
    df_stream = rainbear.scan_zarr(hqb).filter(prd).collect()
    assert df_sync.shape == df_stream.shape
    assert set(df_sync.columns) == set(df_stream.columns)


def test_streaming_small_batch_matches_sync_scan(
    nz_style_point_zarr: str,
) -> None:
    """Join-closed streaming batches must match sync even with a tiny batch_size."""
    hqb = rainbear.ZarrBackendSync.from_url(nz_style_point_zarr)
    prd = pl.col("lead_time") == timedelta(hours=0)
    df_sync = hqb.scan_zarr_sync(predicate=prd)
    batches = list(
        hqb.scan_zarr_streaming_sync(
            predicate=prd,
            batch_size=12,
        ),
    )
    df_stream = pl.concat(batches)
    assert df_sync.shape == df_stream.shape
    assert set(df_sync.columns) == set(df_stream.columns)


@pytest.mark.skipif(
    not _REPO_NZ_ZARR.is_dir(),
    reason="repo sandbox/nz.zarr not present",
)
def test_streaming_filter_select_schema() -> None:
    """Real nz-style zarr: vlen string ``station_id`` on ``point`` must survive filter+select."""
    path = str(_REPO_NZ_ZARR)
    lf = (
        rainbear.scan_zarr(path)
        .filter(
            (pl.col("time") == datetime(2026, 4, 1, 0))
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
        .select(["rime_ice_probability_1hr", "time", "station_id"])
    )
    schema = lf.collect_schema()
    assert schema.len() == 3
    assert set(schema.keys()) == {"rime_ice_probability_1hr", "time", "station_id"}


@pytest.mark.skipif(
    not _REPO_NZ_ZARR.is_dir(),
    reason="repo sandbox/nz.zarr not present",
)
def test_streaming_filter_select_vlen_string_station_id() -> None:
    """Real nz-style zarr: vlen string ``station_id`` on ``point`` must survive filter+select."""
    path = str(_REPO_NZ_ZARR)
    lf = (
        rainbear.scan_zarr(path)
        .filter(
            (pl.col("time") == datetime(2026, 4, 1, 0))
            & (pl.col("lead_time") == timedelta(hours=1)),
        )
        .select(["rime_ice_probability_1hr", "time", "station_id"])
    )
    df = lf.collect()
    assert df.height == 15
    assert df["station_id"].dtype == pl.String
    assert df["station_id"].null_count() == 0
    assert df["station_id"][0] == "BELAIR_C"
