from __future__ import annotations

from typing import Any

import polars as pl

def print_extension_info() -> str: ...
def selected_chunks(
    zarr_url: str,
    predicate: pl.Expr,
    variables: list[str] | None = None,
) -> list[dict[str, Any]]: ...

def _selected_chunks_debug(
    zarr_url: str,
    predicate: pl.Expr,
    variables: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]: ...

def _selected_variables_debug(
    zarr_url: str,
    expr: pl.Expr,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], int]:
    """Debug function that returns per-variable chunk selections.

    Returns:
        A tuple of:
        - inferred_variables: List of variable names found in the DatasetSelection
        - per_variable_chunks: Dict mapping variable name -> list of chunk dicts
        - coord_reads: Number of coordinate array reads performed
    """
    ...

def _debug_expr_ast(predicate: pl.Expr) -> str:
    """Debug function that returns the parsed expression AST as a string."""
    ...

def _debug_chunk_planning(
    zarr_url: str,
    predicate: pl.Expr,
    primary_var: str | None = None,
) -> dict[str, Any]:
    """Comprehensive debug function for chunk planning.

    Returns a dictionary with:
    - meta: Dataset metadata including time encoding info for each array
    - dims: List of dimensions
    - dim_lengths: Map of dimension name to length
    - expr_ast: Parsed expression AST
    - lazy_selection: String representation of the lazy selection (before resolution)
    - resolution_requests: List of resolution requests that will be made
    - resolution_results: Map of request to result (after resolution)
    - materialized_selection: The final materialized selection
    - coord_reads: Number of coordinate array reads performed
    - error: Any error that occurred (if applicable)
    """
    ...

def _debug_coord_array(
    zarr_url: str,
    dim_name: str,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """Debug function to inspect a coordinate array's values and time encoding.

    Returns a dictionary with:
    - dim_name: The dimension name
    - shape: The array shape
    - time_encoding: Time encoding info (if present)
    - sample_raw_values: First and last few raw values from the array
    - sample_decoded_values: The same values after time encoding is applied
    """
    ...

def _debug_literal_conversion(
    zarr_url: str,
    dim_name: str,
    test_value: Any,
) -> dict[str, Any]:
    """Debug function to test literal conversion.

    Shows how a Python value would be converted to a CoordScalar for comparison.
    """
    ...

def scan_zarr_async(
    zarr_url: str,
    predicate: pl.Expr,
    variables: list[str] | None = None,
    max_concurrency: int | None = None,
    with_columns: list[str] | None = None,
) -> Any: ...

class ZarrSource:
    def __init__(
        self,
        zarr_url: str,
        batch_size: int | None,
        n_rows: int | None,
        variables: list[str] | None = None,
        max_chunks_to_read: int | None = None,
    ) -> None: ...

    def schema(self) -> Any: ...
    def try_set_predicate(self, predicate: pl.Expr) -> None: ...
    def set_with_columns(self, columns: list[str]) -> None: ...
    def next(self) -> pl.DataFrame | None: ...
