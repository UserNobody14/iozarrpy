"""Baseline comparison utilities for testing iozarrpy against xarray."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def scan_via_xarray(
    zarr_path: str,
    *,
    columns: list[str],
) -> pl.LazyFrame:
    """
    Load a Zarr dataset via xarray and convert to a Polars LazyFrame.
    
    This is the "baseline" implementation to compare against iozarrpy.
    Returns a LazyFrame so the same filters can be applied to both.
    """
    import xarray as xr

    ds = xr.open_zarr(zarr_path, consolidated=None)

    # Build a tidy DataFrame from xarray
    # Select only the requested columns as data_vars
    data_vars = {name: ds[name] for name in columns if name in ds}
    ds_out = xr.Dataset(data_vars=data_vars)

    pdf = ds_out.to_dataframe().reset_index()
    df = pl.from_pandas(pdf)

    # Select columns in the requested order
    df = df.select(columns)
    
    return df.lazy()


def assert_frames_equal(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    sort_by: list[str] | None = None,
    drop_nan: bool = True,
) -> None:
    """Assert two Polars DataFrames are equal, optionally sorting first.
    
    Args:
        left: First DataFrame (typically iozarrpy output)
        right: Second DataFrame (typically xarray baseline)
        sort_by: Columns to sort by before comparison
        drop_nan: If True, drop rows with NaN values (to match xarray's dropna behavior)
    """
    if drop_nan:
        # Drop null values
        left = left.drop_nulls()
        right = right.drop_nulls()
        
        # Also drop rows with NaN in float columns
        for col, dtype in zip(left.columns, left.dtypes):
            if dtype in (pl.Float32, pl.Float64):
                left = left.filter(~pl.col(col).is_nan())
        for col, dtype in zip(right.columns, right.dtypes):
            if dtype in (pl.Float32, pl.Float64):
                right = right.filter(~pl.col(col).is_nan())

    if sort_by:
        left = left.sort(sort_by)
        right = right.sort(sort_by)
    
    assert_frame_equal(left, right)
