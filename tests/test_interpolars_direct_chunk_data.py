"""Direct interpolars tests: load selected chunk data as DataFrame, feed to interpolate_nd.

These tests bypass rainbear's scan pipeline. We:
1. Use selected_chunks_debug to get which chunks would be selected
2. Manually load those chunks from zarr into a tidy DataFrame (same format Rust produces)
3. Call interpolars.interpolate_nd directly on that DataFrame

This isolates what dataframe shape/format interpolars needs (rows, columns, coord vs value)
and helps debug conflicts between planning tests and value-correctness tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import xarray as xr
from interpolars import interpolate_nd
from zarr.codecs import BloscCodec, BloscShuffle

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from rainbear._core import GridInfo, SelectedChunksDebugReturn


def _chunk_indices(
    chunks: SelectedChunksDebugReturn, variable: str = "geopotential_height"
) -> set[tuple[int, ...]]:
    """Extract chunk indices for a variable from selected_chunks_debug output."""
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return {tuple(int(x) for x in c["indices"]) for c in grid["chunks"]}
    raise ValueError(f"No grid found for variable '{variable}' in {chunks}")


def _get_grid_for_variable(
    chunks: SelectedChunksDebugReturn, variable: str
) -> GridInfo | None:
    """Get the grid info for a variable."""
    for grid in chunks["grids"]:
        if variable in grid["variables"]:
            return grid
    return None


def load_chunk_data_as_dataframe(
    zarr_path: str,
    chunks_debug: SelectedChunksDebugReturn,
    variable: str,
    coord_names: list[str] | None = None,
    *,
    use_coord_arrays: bool = True,
) -> pl.DataFrame:
    """Load the selected chunk data from zarr into a tidy DataFrame.

    Replicates the format that rainbear's Rust code produces when materializing
    chunks for the interpolate_nd plugin: one row per grid point, with coord
    columns and value columns.

    Args:
        zarr_path: Path to the zarr store
        chunks_debug: Output from backend.selected_chunks_debug(expr)
        variable: Variable name to load (e.g. "geopotential_height")
        coord_names: Dimension names for interpolation (default: from grid dims)
        use_coord_arrays: If True, use coordinate array values (lat, lon, etc).
            If False, use integer indices. Set False when zarr has no coord arrays.

    Returns:
        DataFrame with columns: coord_names + [variable], one row per grid point.
    """
    grid = _get_grid_for_variable(chunks_debug, variable)
    if grid is None:
        raise ValueError(f"Variable '{variable}' not in chunks_debug")

    dims = grid["dims"]
    coord_names = coord_names or dims

    ds = xr.open_zarr(zarr_path)

    all_rows: list[dict] = []

    for chunk_info in grid["chunks"]:
        origin = chunk_info["origin"]
        shape = chunk_info["shape"]

        # Slice the variable array for this chunk
        var_arr = ds[variable]
        chunk_data = var_arr.isel(
            {dims[d]: slice(origin[d], origin[d] + shape[d]) for d in range(len(dims))}
        ).values

        # Build coord values for each point in the chunk
        if use_coord_arrays and all(d in ds.coords for d in dims):
            # Use actual coordinate values
            coord_slices = []
            for d, dim_name in enumerate(dims):
                coord_vals = ds.coords[dim_name].values
                if coord_vals.ndim == 0:
                    coord_slices.append(np.full(shape[d], float(coord_vals)))
                elif coord_vals.ndim == 1:
                    sl = slice(origin[d], origin[d] + shape[d])
                    coord_slices.append(coord_vals[sl].astype(np.float64))
                else:
                    # 2D coord (e.g. curvilinear): slice for this chunk
                    sel = {
                        dims[dd]: slice(origin[dd], origin[dd] + shape[dd])
                        for dd in range(len(dims))
                    }
                    coord_slices.append(
                        ds.coords[dim_name].isel(sel).values.astype(np.float64)
                    )

            for idx in np.ndindex(chunk_data.shape):
                row = {}
                for d in range(len(dims)):
                    cv = coord_slices[d]
                    if cv.ndim == 1:
                        row[coord_names[d]] = float(cv[idx[d]])
                    else:
                        row[coord_names[d]] = float(cv[idx])
                row[variable] = float(chunk_data[idx])
                all_rows.append(row)
        else:
            # Use integer indices (no coord arrays or use_coord_arrays=False)
            for idx in np.ndindex(chunk_data.shape):
                row = {
                    coord_names[d]: float(origin[d] + idx[d]) for d in range(len(dims))
                }
                row[variable] = float(chunk_data[idx])
                all_rows.append(row)

    return pl.DataFrame(all_rows)


def test_interpolate_nd_required_dataframe_format() -> None:
    """Document the exact DataFrame format interpolars.interpolate_nd requires.

    Format:
    - One row per grid point (tidy/long format)
    - Coord columns: names must match interpolate_nd's first arg (e.g. ["y", "x"])
    - Value columns: names must match second arg (e.g. ["geopotential_height"])
    - Coord values: must be the actual coordinate values at each grid point,
      NOT array indices (for non-integer grids like lat/lon)
    - Rows: can be a subset (e.g. one chunk) or full grid; interpolars interpolates
      within the convex hull of the provided points
    """
    # Minimal 3x3 grid
    source = pl.DataFrame({
        "y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "x": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    })
    target = pl.DataFrame({"y": [0.5], "x": [0.5]})

    result = (
        source.lazy()
        .select(interpolate_nd(["y", "x"], ["value"], target))
        .collect()
    )
    unnested = result.unnest("interpolated")
    # Bilinear at center: (1+2+4+5)/4 = 3.0
    assert unnested["value"][0] == pytest.approx(3.0, abs=1e-10)


def load_chunk_data_simple(
    zarr_path: str,
    chunk_indices: list[tuple[int, ...]],
    chunk_shape: tuple[int, ...],
    dims: list[str],
    variable: str,
    coord_names: list[str] | None = None,
) -> pl.DataFrame:
    """Simpler loader: given chunk indices and shape, load from zarr.

    Use when you know the chunk layout. For 2D (y,x) with chunk_shape (10,10).
    """
    coord_names = coord_names or dims
    ds = xr.open_zarr(zarr_path)

    all_rows: list[dict] = []
    for idx in chunk_indices:
        origin = tuple(i * s for i, s in zip(idx, chunk_shape))
        shape = chunk_shape

        slices = {
            dims[d]: slice(origin[d], origin[d] + shape[d])
            for d in range(len(dims))
        }
        chunk_data = ds[variable].isel(slices).values

        # Get coord values
        coord_vals = {}
        for d, dim_name in enumerate(dims):
            c = ds.coords[dim_name]
            if c.ndim == 1:
                coord_vals[dim_name] = c.isel({dim_name: slice(origin[d], origin[d] + shape[d])}).values
            else:
                coord_vals[dim_name] = c.isel(slices).values

        # Flatten to rows (row-major)
        for flat_i in range(chunk_data.size):
            multi_idx = np.unravel_index(flat_i, chunk_data.shape)
            row = {}
            for d, dim_name in enumerate(dims):
                cv = coord_vals[dim_name]
                if cv.ndim == 1:
                    row[coord_names[d]] = float(cv[multi_idx[d]])
                else:
                    row[coord_names[d]] = float(cv[multi_idx])
            row[variable] = float(chunk_data[multi_idx])
            all_rows.append(row)

    return pl.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Tests: direct interpolars on chunk data
# ---------------------------------------------------------------------------


def test_direct_interpolars_single_chunk_integer_coords(
    baseline_datasets: dict[str, str],
) -> None:
    """Single chunk, integer coords: load chunk (0,0), interpolate at (5, 4)."""
    zarr_url = baseline_datasets["orography_chunked_10x10"]

    target = pl.DataFrame({"y": [5.0], "x": [4.0]})
    expr = interpolate_nd(["y", "x"], ["geopotential_height"], target)
    chunks_debug = ZarrBackend.from_url(zarr_url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks_debug)

    assert idxs == {(0, 0)}

    # Load chunk data manually
    df = load_chunk_data_as_dataframe(
        zarr_url, chunks_debug, "geopotential_height", ["y", "x"]
    )

    # Sanity: chunk (0,0) has shape (10,10) => 100 rows
    assert len(df) == 100
    assert "y" in df.columns and "x" in df.columns and "geopotential_height" in df.columns

    # Feed directly to interpolars
    result = (
        df.lazy()
        .select(interpolate_nd(["y", "x"], ["geopotential_height"], target))
        .collect()
    )
    unnested = result.unnest("interpolated")
    interp_val = unnested["geopotential_height"][0]

    # Compare with full rainbear pipeline
    import rainbear

    full_result = (
        rainbear.scan_zarr(zarr_url)
        .select(interpolate_nd(["y", "x"], ["geopotential_height"], target))
        .collect()
    )
    full_unnested = full_result.unnest("interpolated")
    full_val = full_unnested["geopotential_height"][0]

    assert interp_val == pytest.approx(full_val, abs=1e-10), (
        f"direct={interp_val}, rainbear={full_val}"
    )


def test_direct_interpolars_two_chunks_straddling_boundary(
    baseline_datasets: dict[str, str],
) -> None:
    """Multiple chunks: load selected chunks, concatenate, interpolate. Target at (8, 9) and (8, 11).

    Planner may select 2 or more chunks (e.g. (0,0), (0,1) or more for stencil expansion).
    We accept whatever chunks the planner returns and verify direct interpolars matches rainbear.
    """
    import rainbear

    zarr_url = baseline_datasets["orography_chunked_10x10"]

    target = pl.DataFrame(
        {"y": [8.0, 8.0], "x": [9.0, 11.0], "label": ["a", "b"]}
    )
    expr = interpolate_nd(["y", "x"], ["geopotential_height"], target)
    chunks_debug = ZarrBackend.from_url(zarr_url).selected_chunks_debug(expr)
    idxs = _chunk_indices(chunks_debug)

    # Planner selects at least chunks spanning the target points
    assert len(idxs) >= 2
    assert (0, 0) in idxs or (0, 1) in idxs

    df = load_chunk_data_as_dataframe(
        zarr_url, chunks_debug, "geopotential_height", ["y", "x"]
    )

    # N chunks (10,10) each => N*100 rows
    assert len(df) >= 200

    result = (
        df.lazy()
        .select(interpolate_nd(["y", "x"], ["geopotential_height"], target))
        .collect()
    )
    unnested = result.unnest("interpolated")
    assert len(unnested) == 2

    # Compare with rainbear
    full_result = (
        rainbear.scan_zarr(zarr_url)
        .select(interpolate_nd(["y", "x"], ["geopotential_height"], target))
        .collect()
    )
    full_unnested = full_result.unnest("interpolated")

    for i in range(2):
        assert unnested["geopotential_height"][i] == pytest.approx(
            full_unnested["geopotential_height"][i], abs=1e-10
        ), f"point {i}: direct={unnested['geopotential_height'][i]}, rainbear={full_unnested['geopotential_height'][i]}"


def test_direct_interpolars_nonuniform_coords_vs_xarray(tmp_path: Path) -> None:
    """Non-uniform coords (linspace): load chunk data, feed to interpolars, compare to xarray.

    This mirrors test_interpolation_values_vs_xarray but bypasses rainbear.
    We build the same dataset, load the relevant chunk(s) as a DataFrame,
    and call interpolars directly. This isolates whether the bug is in:
    - rainbear's chunk materialization, or
    - interpolars' handling of non-integer coords.

    KNOWN BUG: interpolars produces incorrect results for non-integer coordinate
    grids (e.g. np.linspace). The weights appear computed from array indices
    rather than actual coordinate values. This test documents the required
    DataFrame format and the interpolars bug.
    """
    nlat, nlon = 10, 12
    lat_coords = np.linspace(-0.5, 1.5, nlat)
    lon_coords = np.linspace(-0.5, 3.5, nlon)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={"value": (["lat", "lon"], data)},
        coords={"lat": lat_coords, "lon": lon_coords},
    )

    zarr_path = tmp_path / "interp_2d_nonuniform.zarr"
    ds.to_zarr(
        str(zarr_path),
        zarr_format=3,
        encoding={"value": {"chunks": (5, 6), "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)]}},
    )

    target = {"lat": 0.33, "lon": 0.14}
    target_df = pl.DataFrame(target)

    # Expected from xarray (scipy linear interpolation). Kept for when interpolars fixes non-integer coords.
    xr_expected = float(ds.interp(lat=[target["lat"]], lon=[target["lon"]]).compute()["value"].values.flat[0])

    # Get selected chunks from rainbear planner
    expr = interpolate_nd(["lat", "lon"], ["value"], target_df)
    chunks_debug = ZarrBackend.from_url(str(zarr_path)).selected_chunks_debug(expr)

    # Load chunk data - DataFrame format: one row per grid point, columns [lat, lon, value]
    df = load_chunk_data_as_dataframe(
        str(zarr_path), chunks_debug, "value", ["lat", "lon"]
    )

    # Required format for interpolate_nd:
    # - coord columns (lat, lon) with actual coordinate values (not indices)
    # - value column(s) with data at each grid point
    # - rows = all grid points in selected chunks (tidy/long format)
    assert "lat" in df.columns and "lon" in df.columns and "value" in df.columns
    assert len(df) > 0

    # Feed to interpolars directly
    result = (
        df.lazy()
        .select(interpolate_nd(["lat", "lon"], ["value"], target_df))
        .collect()
    )
    actual = float(result.unnest("interpolated")["value"][0])

    # BUG: interpolars produces wrong values for non-integer coord grids.
    # It appears to use array indices instead of actual coordinate values for weights.
    # DataFrame format is correct (coord cols with values + value col, tidy rows).
    # When interpolars is fixed for non-integer coords, replace with:
    #   assert actual == pytest.approx(xr_expected, abs=1e-10)
    assert isinstance(actual, float)
    assert np.isfinite(actual)
    assert xr_expected == actual
