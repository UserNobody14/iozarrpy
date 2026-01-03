"""Baseline comparison utilities for testing iozarrpy against xarray."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal


def xarray_zarr_to_polars_tidy(
    zarr_path: str,
    *,
    columns: list[str],
    selection: Callable[[Any], Any] | None = None,
    dropna: bool = True,
) -> pl.DataFrame:
    """
    Baseline ("old-fashioned") implementation:
    - load via xarray
    - apply selections using xarray semantics
    - convert to a "tidy" table via .to_dataframe().reset_index()
    - convert to Polars and select the intended schema columns
    """
    import xarray as xr

    ds = xr.open_zarr(zarr_path, consolidated=None)
    if selection is not None:
        ds = selection(ds)

    # Ensure requested columns exist as variables in a Dataset.
    # This supports both data_vars and coords (coords become data_vars in this view).
    data_vars = {name: ds[name] for name in columns if name in ds}
    ds_out = xr.Dataset(data_vars=data_vars)

    pdf = ds_out.to_dataframe().reset_index()
    if dropna:
        pdf = pdf.dropna()

    df = pl.from_pandas(pdf)

    # Enforce final schema / column ordering
    return df.select(columns)


def assert_frames_equal(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    sort_by: list[str] | None = None,
) -> None:
    """Assert two Polars DataFrames are equal, optionally sorting first."""
    if sort_by:
        left = left.sort(sort_by)
        right = right.sort(sort_by)
    assert_frame_equal(left, right)
