"""Tests for zarr arrays with string and binary dtypes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest

import rainbear


@pytest.fixture
def string_dataset(output_dir: Path) -> str:
    """Create a zarr dataset with a string variable alongside a numeric one, using xarray."""
    import xarray as xr

    path = output_dir / "string_dtype.zarr"
    if path.exists():
        shutil.rmtree(path)

    y = np.arange(4, dtype=np.int64)
    x = np.arange(3, dtype=np.int64)
    # NumPy unicode dtype (<U#) becomes Zarr v3 fixed_length_utf32, which rainbear
    # does not read. NumPy 2 StringDType is written as data_type: string (UTF-8).
    label = np.array(
        [
            ["a0", "a1", "a2"],
            ["b0", "b1", "b2"],
            ["c0", "c1", "c2"],
            ["d0", "d1", "d2"],
        ],
        dtype=np.dtypes.StringDType(),
    )
    value = np.arange(12, dtype=np.float64).reshape(4, 3)

    ds = xr.Dataset(
        {
            "label": (["y", "x"], label),
            "value": (["y", "x"], value),
        },
        coords={
            "y": ("y", y),
            "x": ("x", x),
        },
    )

    # Write as Zarr v3 with chunking
    ds.to_zarr(
        str(path),
        mode="w",
        encoding={
            "label": {"chunks": (2, 2)},
            "value": {"chunks": (2, 2)},
            "y": {"chunks": (2,)},
            "x": {"chunks": (2,)},
        },
        consolidated=False,
        zarr_version=3,
    )

    return str(path)


def test_scan_string_smoke(string_dataset: str) -> None:
    """Basic scan of a dataset containing a string variable."""
    lf = rainbear.scan_zarr(string_dataset)
    df = lf.collect()

    assert df.shape == (12, 4)
    assert set(df.columns) == {"y", "x", "label", "value"}
    assert df["label"].dtype == pl.String
    assert df["value"].dtype == pl.Float64


def test_string_filter_on_coordinate(string_dataset: str) -> None:
    """Filter on an integer coordinate; string column should be preserved."""
    lf = rainbear.scan_zarr(string_dataset)
    df = lf.filter(pl.col("y") >= 2).collect()

    assert df.shape == (6, 4)
    # All remaining y values should be >= 2
    assert cast(int, df["y"].min()) >= 2
    # Verify the string values match expectations
    labels = set(df["label"].to_list())
    assert labels == {"c0", "c1", "c2", "d0", "d1", "d2"}


def test_string_select_columns(string_dataset: str) -> None:
    """Select only specific columns including the string variable."""
    df = (
        rainbear.scan_zarr(string_dataset)
        .select(["y", "x", "label"])
        .collect()
    )

    assert df.shape == (12, 3)
    assert df.columns == ["y", "x", "label"]
    assert df["label"].dtype == pl.String


def test_string_values_correct(string_dataset: str) -> None:
    """Verify that all string values are read correctly."""
    df = rainbear.scan_zarr(string_dataset).collect().sort(["y", "x"])

    expected_labels = [
        "a0", "a1", "a2",
        "b0", "b1", "b2",
        "c0", "c1", "c2",
        "d0", "d1", "d2",
    ]
    assert df["label"].to_list() == expected_labels


def test_string_post_filter(string_dataset: str) -> None:
    """Collect then filter on string column with Polars."""
    df = (
        rainbear.scan_zarr(string_dataset)
        .collect()
        .filter(pl.col("label") == "b1")
    )

    assert df.shape == (1, 4)
    assert df["y"].item() == 1
    assert df["x"].item() == 1
    assert df["value"].item() == 4.0
