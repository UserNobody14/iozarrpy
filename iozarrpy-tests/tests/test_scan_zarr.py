import shutil
import unittest
from pathlib import Path

import polars as pl

import iozarrpy
from iozarrpy import _core


def dataset_path(name: str) -> str:
    out_dir = Path(__file__).resolve().parent / "output-datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if path.exists():
        shutil.rmtree(path)
    return str(path)


class TestScanZarr(unittest.TestCase):
    def test_scan_zarr_smoke(self) -> None:
        path = dataset_path("demo_store.zarr")
        _core._create_demo_store(path)

        lf = iozarrpy.scan_zarr(path, size=1000)
        df = lf.collect()

        self.assertEqual(df.height, 12)
        self.assertEqual(df.columns, ["time", "lat", "temp"])

    def test_sel_predicate(self) -> None:
        path = dataset_path("demo_store_sel.zarr")
        _core._create_demo_store(path)

        lf = iozarrpy.scan_zarr(path, size=1000)
        lf = lf.sel((pl.col("lat") >= 20.0) & (pl.col("lat") <= 30.0))
        df = lf.collect()

        self.assertEqual(df.height, 8)
        self.assertTrue(df["lat"].is_in([20.0, 30.0]).all())


if __name__ == "__main__":
    unittest.main()

