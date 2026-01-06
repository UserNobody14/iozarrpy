"""End-to-end tests for generated orography datasets."""

from __future__ import annotations

import polars as pl

import rainbear

from . import zarr_generators


def test_generated_orography_scan(dataset_path) -> None:
    # Keep this small so it runs fast in CI/dev.
    ds = zarr_generators.create_orography_dataset(nx=32, ny=24, sigma=5.0, seed=123)

    path = dataset_path("orography_small.zarr")
    ds.to_zarr(path, zarr_format=3)

    lf = rainbear.scan_zarr(path, variables=["geopotential_height"])
    df = lf.collect()

    assert df.columns == ["y", "x", "geopotential_height"]
    assert df.height == 32 * 24


def test_generated_orography_sel(dataset_path) -> None:
    ds = zarr_generators.create_orography_dataset(nx=20, ny=10, sigma=4.0, seed=7)

    path = dataset_path("orography_sel.zarr")
    ds.to_zarr(path, zarr_format=3)

    lf = rainbear.scan_zarr(path, variables=["geopotential_height"])
    lf = lf.filter((pl.col("y") >= 3) & (pl.col("y") <= 6))
    df = lf.collect()

    # y in [3..6] => 4 y-values, x has 20 values
    assert df.height == 4 * 20
    assert df["y"].is_between(3, 6).all()


def test_generated_orography_multi_var(dataset_path) -> None:
    # Uses 2D vars (latitude/longitude) + geopotential_height on dims (y, x)
    ds = zarr_generators.create_orography_dataset(nx=12, ny=9, sigma=3.0, seed=99)

    path = dataset_path("orography_multi_var.zarr")
    ds.to_zarr(path, zarr_format=3)

    lf = rainbear.scan_zarr(
        path,
        variables=["geopotential_height", "latitude", "longitude"],
    )
    df = lf.collect()
    assert df.columns == ["y", "x", "geopotential_height", "latitude", "longitude"]
    assert df.height == 12 * 9


def test_generated_orography_unconsolidated(dataset_path) -> None:
    ds = zarr_generators.create_orography_dataset(nx=16, ny=8, sigma=4.0, seed=5)

    path = dataset_path("orography_unconsolidated.zarr")
    ds.to_zarr(path, zarr_format=3, consolidated=False)

    lf = rainbear.scan_zarr(path, variables=["geopotential_height"])
    df = lf.collect()
    assert df.height == 16 * 8


def test_generated_orography_projection(dataset_path) -> None:
    ds = zarr_generators.create_orography_dataset(nx=10, ny=6, sigma=3.0, seed=1)

    path = dataset_path("orography_projection.zarr")
    ds.to_zarr(path, zarr_format=3)

    # Ensure projection pushdown works (only variable column emitted).
    lf = rainbear.scan_zarr(path, variables=["geopotential_height"])
    df = lf.select("geopotential_height").collect()
    assert df.columns == ["geopotential_height"]
    assert df.height == 10 * 6
    assert df.height == 10 * 6
    assert df.height == 10 * 6
