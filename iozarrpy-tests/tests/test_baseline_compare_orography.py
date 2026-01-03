"""Tests comparing iozarrpy output against xarray baseline across many dataset configurations."""

from __future__ import annotations

import polars as pl
import pytest

import iozarrpy
from tests.baseline_utils import assert_frames_equal, xarray_zarr_to_polars_tidy
from tests.conftest import BASELINE_DATASET_CONFIGS, get_dataset_config

# Generate test IDs from dataset names
DATASET_NAMES = [cfg.name for cfg in BASELINE_DATASET_CONFIGS]


@pytest.mark.parametrize("dataset_name", DATASET_NAMES)
def test_baseline_equals_scan_no_selection(baseline_datasets: dict[str, str], dataset_name: str) -> None:
    """Test that scanning a zarr without selection matches xarray baseline."""
    cfg = get_dataset_config(dataset_name)
    path = baseline_datasets[dataset_name]

    # iozarrpy (system under test)
    out = (
        iozarrpy.scan_zarr(path, variables=cfg.variables, size=1_000_000)
        .collect()
        .select(cfg.expected_cols)
    )

    # Baseline (xarray -> polars conversion)
    baseline = xarray_zarr_to_polars_tidy(path, columns=cfg.expected_cols)

    assert_frames_equal(out, baseline, sort_by=cfg.dims)


@pytest.mark.parametrize("dataset_name", DATASET_NAMES)
def test_baseline_equals_scan_with_dim_filter(baseline_datasets: dict[str, str], dataset_name: str) -> None:
    """Test that scanning with a dimension filter matches xarray baseline."""
    cfg = get_dataset_config(dataset_name)
    path = baseline_datasets[dataset_name]

    if cfg.filter_dim is None or cfg.filter_range is None:
        pytest.skip(f"Dataset {dataset_name} has no filter configuration")

    dim = cfg.filter_dim
    dim_min, dim_max = cfg.filter_range

    # iozarrpy selection (Polars)
    lf = iozarrpy.scan_zarr(path, variables=cfg.variables, size=1_000_000)
    lf = lf.sel((pl.col(dim) >= dim_min) & (pl.col(dim) <= dim_max))
    out = lf.collect().select(cfg.expected_cols)

    # Baseline selection (xarray)
    def selection(ds):
        cond = (ds[dim] >= dim_min) & (ds[dim] <= dim_max)
        return ds.where(cond, drop=True)

    baseline = xarray_zarr_to_polars_tidy(path, columns=cfg.expected_cols, selection=selection)

    assert_frames_equal(out, baseline, sort_by=cfg.dims)


# Datasets that have 2D coords (latitude/longitude) for complex filter tests
DATASETS_WITH_2D_COORDS = [
    cfg.name for cfg in BASELINE_DATASET_CONFIGS
    if "latitude" in cfg.expected_cols or any("latitude" in v for v in cfg.variables)
]


@pytest.mark.parametrize("dataset_name", DATASETS_WITH_2D_COORDS)
def test_baseline_equals_scan_filter_on_2d_coords(
    baseline_datasets: dict[str, str], dataset_name: str
) -> None:
    """Test that filters on 2D coordinates match xarray baseline."""
    cfg = get_dataset_config(dataset_name)
    path = baseline_datasets[dataset_name]

    # Modify config to include latitude/longitude if not already
    variables = list(cfg.variables)
    expected_cols = list(cfg.expected_cols)
    if "latitude" not in variables:
        variables.append("latitude")
    if "longitude" not in variables:
        variables.append("longitude")
    if "latitude" not in expected_cols:
        expected_cols.append("latitude")
    if "longitude" not in expected_cols:
        expected_cols.append("longitude")

    # Use filter values that will produce some results for most datasets
    # These are based on the synthetic lon/lat grid: lon = -130 + x*0.02, lat = 20 + y*0.02
    lat_min, lat_max = 20.04, 20.14
    lon_min, lon_max = -129.96, -129.86

    # iozarrpy selection (Polars)
    lf = iozarrpy.scan_zarr(path, variables=variables, size=1_000_000)
    lf = lf.sel(
        (pl.col("latitude") >= lat_min)
        & (pl.col("latitude") <= lat_max)
        & (pl.col("longitude") >= lon_min)
        & (pl.col("longitude") <= lon_max)
    )
    out = lf.collect().select(expected_cols)

    # Baseline selection (xarray)
    def selection(ds):
        cond = (
            (ds["latitude"] >= lat_min)
            & (ds["latitude"] <= lat_max)
            & (ds["longitude"] >= lon_min)
            & (ds["longitude"] <= lon_max)
        )
        return ds.where(cond, drop=True)

    baseline = xarray_zarr_to_polars_tidy(path, columns=expected_cols, selection=selection)

    assert_frames_equal(out, baseline, sort_by=cfg.dims)


# Sharded datasets
SHARDED_DATASETS = [
    cfg.name for cfg in BASELINE_DATASET_CONFIGS
    if "sharded" in cfg.name
]


@pytest.mark.parametrize("dataset_name", SHARDED_DATASETS)
def test_baseline_equals_scan_sharded_store(baseline_datasets: dict[str, str], dataset_name: str) -> None:
    """Test that sharded zarr stores match xarray baseline (explicit sharding test)."""
    cfg = get_dataset_config(dataset_name)
    path = baseline_datasets[dataset_name]

    # iozarrpy (system under test)
    out = iozarrpy.scan_zarr(path, variables=cfg.variables, size=1_000_000).collect()
    out = out.select(cfg.expected_cols)

    # Baseline
    baseline = xarray_zarr_to_polars_tidy(path, columns=cfg.expected_cols)

    assert_frames_equal(out, baseline, sort_by=cfg.dims)


# Big endian datasets
BIG_ENDIAN_DATASETS = [
    cfg.name for cfg in BASELINE_DATASET_CONFIGS
    if "_big" in cfg.name
]


@pytest.mark.parametrize("dataset_name", BIG_ENDIAN_DATASETS)
def test_baseline_equals_scan_big_endian(baseline_datasets: dict[str, str], dataset_name: str) -> None:
    """Test that big-endian encoded data matches xarray baseline."""
    cfg = get_dataset_config(dataset_name)
    path = baseline_datasets[dataset_name]

    # iozarrpy (system under test)
    out = iozarrpy.scan_zarr(path, variables=cfg.variables, size=1_000_000).collect()
    out = out.select(cfg.expected_cols)

    # Baseline
    baseline = xarray_zarr_to_polars_tidy(path, columns=cfg.expected_cols)

    assert_frames_equal(out, baseline, sort_by=cfg.dims)
