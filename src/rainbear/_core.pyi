from __future__ import annotations

from typing import Any, Protocol

import polars as pl

# Type for ObjectStore instances (from rainbear.store or obstore)
class ObjectStore(Protocol):
    """Protocol for ObjectStore instances."""
    ...

# Type alias for store input - can be a URL string or an ObjectStore instance
StoreInput = str | ObjectStore

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
    store: StoreInput,
    predicate: pl.Expr,
    variables: list[str] | None = None,
    max_concurrency: int | None = None,
    with_columns: list[str] | None = None,
    prefix: str | None = None,
) -> Any:
    """Async scan a Zarr store and return a DataFrame.
    
    Args:
        store: Either a URL string or an ObjectStore instance.
        predicate: Filter expression to apply.
        variables: Optional list of variable names to read.
        max_concurrency: Maximum number of concurrent chunk reads.
        with_columns: Optional list of columns to read.
        prefix: Optional path prefix within the store (for ObjectStore instances).
    """
    ...

class ZarrSource:
    """Low-level Zarr source for streaming chunk reads."""
    
    def __init__(
        self,
        store: StoreInput,
        batch_size: int | None,
        n_rows: int | None,
        variables: list[str] | None = None,
        max_chunks_to_read: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """Create a new ZarrSource.
        
        Args:
            store: Either a URL string or an ObjectStore instance.
            batch_size: Number of rows per batch.
            n_rows: Maximum total rows to read.
            variables: Optional list of variable names to read.
            max_chunks_to_read: Optional limit on chunks to read.
            prefix: Optional path prefix within the store (for ObjectStore instances).
        """
        ...

    def schema(self) -> Any: ...
    def try_set_predicate(self, predicate: pl.Expr) -> None: ...
    def set_with_columns(self, columns: list[str]) -> None: ...
    def next(self) -> pl.DataFrame | None: ...


# Store module - provides ObjectStore builders with full connection pooling
class store:
    """Object store builders for S3, GCS, Azure, HTTP, and local filesystem.
    
    Stores created from this module get full connection pooling when used
    with rainbear's scan functions.
    """
    
    class S3Store:
        """Amazon S3 object store."""
        def __init__(
            self,
            bucket: str | None = None,
            *,
            region: str | None = None,
            access_key_id: str | None = None,
            secret_access_key: str | None = None,
            session_token: str | None = None,
            endpoint: str | None = None,
            **kwargs: Any,
        ) -> None: ...
    
    class GCSStore:
        """Google Cloud Storage object store."""
        def __init__(
            self,
            bucket: str | None = None,
            *,
            service_account_path: str | None = None,
            **kwargs: Any,
        ) -> None: ...
    
    class AzureStore:
        """Azure Blob Storage object store."""
        def __init__(
            self,
            container: str | None = None,
            *,
            account: str | None = None,
            access_key: str | None = None,
            **kwargs: Any,
        ) -> None: ...
    
    class HTTPStore:
        """HTTP/HTTPS object store."""
        def __init__(
            self,
            url: str,
            **kwargs: Any,
        ) -> None: ...
    
    class LocalStore:
        """Local filesystem object store."""
        def __init__(
            self,
            prefix: str | None = None,
        ) -> None: ...
    
    class MemoryStore:
        """In-memory object store."""
        def __init__(self) -> None: ...


# Exceptions module
class exceptions:
    """Object store exceptions."""
    
    class ObjectStoreError(Exception):
        """Base exception for object store errors."""
        ...
    
    class NotFoundError(ObjectStoreError):
        """Object not found."""
        ...
    
    class PermissionDeniedError(ObjectStoreError):
        """Permission denied."""
        ...
