import importlib.util
import shutil
import unittest
from pathlib import Path

import polars as pl
from baseline_utils import assert_frames_equal, xarray_zarr_to_polars_tidy

import iozarrpy


def dataset_path(name: str) -> str:
    out_dir = Path(__file__).resolve().parent / "output-datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if path.exists():
        shutil.rmtree(path)
    return str(path)


def load_zarr_gen_test():
    here = Path(__file__).resolve().parent
    target = here / "zarr_gen_test.py"
    spec = importlib.util.spec_from_file_location("zarr_gen_test", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import generator module at {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBaselineCompareOrography(unittest.TestCase):
    def test_baseline_equals_scan_no_selection(self) -> None:
        # Intended schema (both frames):
        # ["y", "x", "geopotential_height", "latitude", "longitude"]
        expected_cols = ["y", "x", "geopotential_height", "latitude", "longitude"]

        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=18, ny=12, sigma=4.0, seed=101)

        path = dataset_path("baseline_orography_no_sel.zarr")
        ds.to_zarr(path, zarr_format=3)

        # iozarrpy (system under test)
        out = (
            iozarrpy.scan_zarr(path, variables=["geopotential_height", "latitude", "longitude"], size=1_000_000)
            .collect()
            .select(expected_cols)
        )

        # Baseline (xarray selection + manual polars conversion)
        baseline = xarray_zarr_to_polars_tidy(path, columns=expected_cols)

        assert_frames_equal(out, baseline, sort_by=["y", "x"])

    def test_baseline_equals_scan_complex_filter_on_2d_coords(self) -> None:
        # Intended schema (both frames):
        # ["y", "x", "geopotential_height", "latitude", "longitude"]
        expected_cols = ["y", "x", "geopotential_height", "latitude", "longitude"]

        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=24, ny=16, sigma=5.0, seed=11)

        path = dataset_path("baseline_orography_complex_sel.zarr")
        ds.to_zarr(path, zarr_format=3)

        lat_min, lat_max = 20.10, 20.20
        lon_min, lon_max = -129.90, -129.80
        y_min, y_max = 3, 10

        # iozarrpy selection (Polars)
        lf = iozarrpy.scan_zarr(path, variables=["geopotential_height", "latitude", "longitude"], size=1_000_000)
        lf = lf.sel(
            (pl.col("y") >= y_min)
            & (pl.col("y") <= y_max)
            & (pl.col("latitude") >= lat_min)
            & (pl.col("latitude") <= lat_max)
            & (pl.col("longitude") >= lon_min)
            & (pl.col("longitude") <= lon_max)
        )
        out = lf.collect().select(expected_cols)

        # Baseline selection (xarray)
        def selection(ds):
            cond = (
                (ds["y"] >= y_min)
                & (ds["y"] <= y_max)
                & (ds["latitude"] >= lat_min)
                & (ds["latitude"] <= lat_max)
                & (ds["longitude"] >= lon_min)
                & (ds["longitude"] <= lon_max)
            )
            return ds.where(cond, drop=True)

        baseline = xarray_zarr_to_polars_tidy(path, columns=expected_cols, selection=selection)

        assert_frames_equal(out, baseline, sort_by=["y", "x"])

    def test_baseline_equals_scan_sharded_store(self) -> None:
        # Intended schema (both frames):
        # ["y", "x", "geopotential_height"]
        expected_cols = ["y", "x", "geopotential_height"]

        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=20, ny=14, sigma=4.0, seed=22)

        path = dataset_path("baseline_orography_sharded.zarr")

        # Sharding/chunking: keep small so it runs fast but exercises zarr v3 sharding.
        encoding = {
            "geopotential_height": {"chunks": (5, 5), "shards": (10, 10)},
        }
        ds[["geopotential_height"]].to_zarr(path, zarr_format=3, encoding=encoding)

        out = iozarrpy.scan_zarr(path, variables=["geopotential_height"], size=1_000_000).collect()
        out = out.select(expected_cols)

        baseline = xarray_zarr_to_polars_tidy(path, columns=expected_cols)

        assert_frames_equal(out, baseline, sort_by=["y", "x"])


