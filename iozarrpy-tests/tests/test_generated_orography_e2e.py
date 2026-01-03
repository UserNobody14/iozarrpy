import importlib.util
import shutil
import unittest
from pathlib import Path

import polars as pl

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


class TestGeneratedOrographyE2E(unittest.TestCase):
    def test_generated_orography_scan(self) -> None:
        # Keep this small so it runs fast in CI/dev.
        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=32, ny=24, sigma=5.0, seed=123)

        path = dataset_path("orography_small.zarr")
        ds.to_zarr(path, zarr_format=3)

        lf = iozarrpy.scan_zarr(path, variables=["geopotential_height"], size=1_000_000)
        df = lf.collect()

        self.assertEqual(df.columns, ["y", "x", "geopotential_height"])
        self.assertEqual(df.height, 32 * 24)

    def test_generated_orography_sel(self) -> None:
        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=20, ny=10, sigma=4.0, seed=7)

        path = dataset_path("orography_sel.zarr")
        ds.to_zarr(path, zarr_format=3)

        lf = iozarrpy.scan_zarr(path, variables=["geopotential_height"], size=1_000_000)
        lf = lf.sel((pl.col("y") >= 3) & (pl.col("y") <= 6))
        df = lf.collect()

        # y in [3..6] => 4 y-values, x has 20 values
        self.assertEqual(df.height, 4 * 20)
        self.assertTrue(df["y"].is_between(3, 6).all())

    def test_generated_orography_multi_var(self) -> None:
        # Uses 2D vars (latitude/longitude) + geopotential_height on dims (y, x)
        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=12, ny=9, sigma=3.0, seed=99)

        path = dataset_path("orography_multi_var.zarr")
        ds.to_zarr(path, zarr_format=3)

        lf = iozarrpy.scan_zarr(
            path,
            variables=["geopotential_height", "latitude", "longitude"],
            size=1_000_000,
        )
        df = lf.collect()
        self.assertEqual(df.columns, ["y", "x", "geopotential_height", "latitude", "longitude"])
        self.assertEqual(df.height, 12 * 9)

    def test_generated_orography_unconsolidated(self) -> None:
        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=16, ny=8, sigma=4.0, seed=5)

        path = dataset_path("orography_unconsolidated.zarr")
        ds.to_zarr(path, zarr_format=3, consolidated=False)

        lf = iozarrpy.scan_zarr(path, variables=["geopotential_height"], size=1_000_000)
        df = lf.collect()
        self.assertEqual(df.height, 16 * 8)

    def test_generated_orography_projection(self) -> None:
        zarr_gen_test = load_zarr_gen_test()
        ds = zarr_gen_test.create_hrrr_orography_dataset(nx=10, ny=6, sigma=3.0, seed=1)

        path = dataset_path("orography_projection.zarr")
        ds.to_zarr(path, zarr_format=3)

        # Ensure projection pushdown works (only variable column emitted).
        lf = iozarrpy.scan_zarr(path, variables=["geopotential_height"], size=1_000_000)
        df = lf.select("geopotential_height").collect()
        self.assertEqual(df.columns, ["geopotential_height"])
        self.assertEqual(df.height, 10 * 6)


if __name__ == "__main__":
    unittest.main()


