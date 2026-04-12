"""Regression: 1D coord dim `a`, 1D var `b(a)`, and 3D `c(foo,bar,a)` read/join correctly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

import rainbear


@pytest.fixture
def dim_a_aux_b_3dc_path(tmp_path: Path) -> str:
    """Flat zarr: coords foo, bar, a; b indexed only by a; c is 3D on foo×bar×a."""
    nfoo, nbar, na = 2, 3, 4
    foo = np.arange(nfoo, dtype=np.int64)
    bar = np.arange(nbar, dtype=np.int64)
    a = np.arange(na, dtype=np.int64)

    path = tmp_path / "dim_a_aux_b_3dc.zarr"
    path.mkdir(parents=True)
    root = zarr.open_group(str(path), mode="w")

    for name, data, chunks in (
        ("foo", foo, (1,)),
        ("bar", bar, (2,)),
        ("a", a, (2,)),
    ):
        arr = root.create_array(name, data=data, chunks=chunks)
        arr.attrs["_ARRAY_DIMENSIONS"] = [name]

    b_data = (np.arange(na, dtype=np.int64) + 1) * 10
    b_arr = root.create_array("b", data=b_data, chunks=(2,))
    b_arr.attrs["_ARRAY_DIMENSIONS"] = ["a"]

    c_data = (
        np.arange(nfoo, dtype=np.int64)[:, None, None] * 1000
        + np.arange(nbar, dtype=np.int64)[None, :, None] * 100
        + np.arange(na, dtype=np.int64)[None, None, :]
    )
    c_arr = root.create_array("c", data=c_data, chunks=(1, 1, 2))
    c_arr.attrs["_ARRAY_DIMENSIONS"] = ["foo", "bar", "a"]

    zarr.consolidate_metadata(str(path))
    return str(path)


def test_select_coord_dim_and_aux_b_with_c(dim_a_aux_b_3dc_path: str) -> None:
    lf = rainbear.scan_zarr(dim_a_aux_b_3dc_path).select(
        ["foo", "bar", "a", "b", "c"],
    )
    df = lf.collect()
    assert df.height == 2 * 3 * 4
    bad = df.filter(pl.col("b") != (pl.col("a") + 1) * 10)
    assert bad.is_empty(), bad
    bad_c = df.filter(
        pl.col("c")
        != pl.col("foo") * 1000 + pl.col("bar") * 100 + pl.col("a")
    )
    assert bad_c.is_empty(), bad_c
