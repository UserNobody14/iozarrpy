from __future__ import annotations

from typing import Any, Protocol

import polars as pl
from typing_extensions import TypedDict

# Type for ObjectStore instances (from rainbear.store or obstore)
class ObjectStore(Protocol):
    """Protocol for ObjectStore instances."""
    ...

# Type alias for store input - can be a URL string or an ObjectStore instance
StoreInput = str | ObjectStore

# Selected chunks debug return type
# {'coord_reads': 0,
#  'grids': [{'chunks': [{'indices': [0], 'origin': [0], 'shape': [50]}],
#             'dims': ['a'],
#             'variables': ['a']},
#            {'chunks': [{'indices': [0, 0, 0],
#                         'origin': [0, 0, 0],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 0, 1],
#                         'origin': [0, 0, 10],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 0, 2],
#                         'origin': [0, 0, 20],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 1, 0],
#                         'origin': [0, 10, 0],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 1, 1],
#                         'origin': [0, 10, 10],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 1, 2],
#                         'origin': [0, 10, 20],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 2, 0],
#                         'origin': [0, 20, 0],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 2, 1],
#                         'origin': [0, 20, 10],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 2, 2],
#                         'origin': [0, 20, 20],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 3, 0],
#                         'origin': [0, 30, 0],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 3, 1],
#                         'origin': [0, 30, 10],
#                         'shape': [10, 10, 10]},
#                        {'indices': [0, 3, 2],
#                         'origin': [0, 30, 20],
#                         'shape': [10, 10, 10]}],
#             'dims': ['a', 'b', 'c'],
#             'variables': ['precip', 'pressure', 'temp', 'wind_u', 'wind_v']},
#            {'chunks': [{'indices': [0], 'origin': [0], 'shape': [40]}],
#             'dims': ['b'],
#             'variables': ['b']},
#            {'chunks': [{'indices': [0, 0], 'origin': [0, 0], 'shape': [10, 10]},
#                        {'indices': [0, 1],
#                         'origin': [0, 10],
#                         'shape': [10, 10]},
#                        {'indices': [0, 2],
#                         'origin': [0, 20],
#                         'shape': [10, 10]},
#                        {'indices': [1, 0],
#                         'origin': [10, 0],
#                         'shape': [10, 10]},
#                        {'indices': [1, 1],
#                         'origin': [10, 10],
#                         'shape': [10, 10]},
#                        {'indices': [1, 2],
#                         'origin': [10, 20],
#                         'shape': [10, 10]},
#                        {'indices': [2, 0],
#                         'origin': [20, 0],
#                         'shape': [10, 10]},
#                        {'indices': [2, 1],
#                         'origin': [20, 10],
#                         'shape': [10, 10]},
#                        {'indices': [2, 2],
#                         'origin': [20, 20],
#                         'shape': [10, 10]},
#                        {'indices': [3, 0],
#                         'origin': [30, 0],
#                         'shape': [10, 10]},
#                        {'indices': [3, 1],
#                         'origin': [30, 10],
#                         'shape': [10, 10]},
#                        {'indices': [3, 2],
#                         'origin': [30, 20],
#                         'shape': [10, 10]}],
#             'dims': ['b', 'c'],
#             'variables': ['surface']},
#            {'chunks': [{'indices': [0], 'origin': [0], 'shape': [30]}],
#             'dims': ['c'],
#             'variables': ['c']}]}

class ChunkInfo(TypedDict):
    indices: list[int]
    origin: list[int]
    shape: list[int]

class GridInfo(TypedDict):
    dims: list[str]
    variables: list[str]
    chunks: list[ChunkInfo]

class SelectedChunksDebugReturn(TypedDict):
    grids: list[GridInfo]
    coord_reads: int

def print_extension_info() -> str: ...

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


class ZarrBackend:
    """Zarr backend with persistent caching across scans.
    
    The backend owns the store and caches coordinate array chunks and metadata
    across multiple scan operations, making repeated queries more efficient.
    
    Examples:
        >>> # Create a backend from URL
        >>> backend = ZarrBackend.from_url("s3://bucket/dataset.zarr")
        >>> 
        >>> # Async scan with caching
        >>> df1 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 1, 1))
        >>> df2 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 6, 1))  # Uses cached coords
        >>>
        >>> # Check cache statistics
        >>> stats = await backend.cache_stats()
        >>> print(f"Cached {stats['coord_entries']} coordinate chunks")
    """
    
    @staticmethod
    def from_url(url: str, max_cache_entries: int = 0) -> ZarrBackend:
        """Create a backend from a URL string.
        
        Args:
            url: URL to the zarr store (e.g., "s3://bucket/path.zarr")
            max_cache_entries: Maximum cached coord chunks (0 = unlimited)
        """
        ...
    
    @staticmethod
    def from_store(
        store: ObjectStore,
        prefix: str | None = None,
        max_cache_entries: int = 0,
    ) -> ZarrBackend:
        """Create a backend from an ObjectStore instance.
        
        Args:
            store: ObjectStore instance (from rainbear.store or obstore)
            prefix: Optional path prefix within the store
            max_cache_entries: Maximum cached coord chunks (0 = unlimited)
        """
        ...
    
    def scan_zarr_async(
        self,
        predicate: pl.Expr,
        variables: list[str] | None = None,
        max_concurrency: int | None = None,
        with_columns: list[str] | None = None,
    ) -> Any:
        """Async scan the zarr store and return a DataFrame.
        
        Uses the backend's cached coordinates for efficient predicate pushdown.
        
        Args:
            predicate: Polars expression for filtering
            variables: Optional list of variable names to read
            max_concurrency: Maximum concurrent chunk reads
            with_columns: Optional list of columns to include
        
        Returns:
            An awaitable that resolves to a pl.DataFrame
        """
        ...

    def selected_chunks_debug(
        self,
        predicate: pl.Expr
    ) -> SelectedChunksDebugReturn: ...
    
    def schema(self, variables: list[str] | None = None) -> Any:
        """Get the schema for the zarr dataset.
        
        Args:
            variables: Optional list of variable names to include
        """
        ...
    
    def root(self) -> str:
        """Get the store root path."""
        ...
    
    def clear_coord_cache(self) -> Any:
        """Clear the coordinate cache (async)."""
        ...
    
    def clear_all_caches(self) -> Any:
        """Clear all caches - metadata and coordinates (async)."""
        ...
    
    def cache_stats(self) -> Any:
        """Get cache statistics (async).
        
        Returns:
            An awaitable that resolves to a dict with:
            - coord_entries: Number of cached coordinate chunks
            - has_metadata: Whether metadata is cached
        """
        ...


class ZarrBackendSync:
    """Zarr backend with persistent caching across scans.
    
    The backend owns the store and caches coordinate array chunks and metadata
    across multiple scan operations, making repeated queries more efficient.
    
    Examples:
        >>> # Create a backend from URL
        >>> backend = ZarrBackend.from_url("s3://bucket/dataset.zarr")
        >>> 
        >>> # Async scan with caching
        >>> df1 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 1, 1))
        >>> df2 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 6, 1))  # Uses cached coords
        >>>
        >>> # Check cache statistics
        >>> stats = await backend.cache_stats()
        >>> print(f"Cached {stats['coord_entries']} coordinate chunks")
    """
    
    @staticmethod
    def from_url(url: str, max_cache_entries: int = 0) -> ZarrBackendSync:
        """Create a backend from a URL string.
        
        Args:
            url: URL to the zarr store (e.g., "s3://bucket/path.zarr")
            max_cache_entries: Maximum cached coord chunks (0 = unlimited)
        """
        ...
    
    @staticmethod
    def from_store(
        store: ObjectStore,
        prefix: str | None = None,
        max_cache_entries: int = 0,
    ) -> ZarrBackendSync:
        """Create a backend from an ObjectStore instance.
        
        Args:
            store: ObjectStore instance (from rainbear.store or obstore)
            prefix: Optional path prefix within the store
            max_cache_entries: Maximum cached coord chunks (0 = unlimited)
        """
        ...
    
    def scan_zarr_sync(
        self,
        predicate: pl.Expr | None = None,
        variables: list[str] | None = None,
        max_concurrency: int | None = None,
        with_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Async scan the zarr store and return a DataFrame.
        
        Uses the backend's cached coordinates for efficient predicate pushdown.
        
        Args:
            predicate: Polars expression for filtering
            variables: Optional list of variable names to read
            max_concurrency: Maximum concurrent chunk reads
            with_columns: Optional list of columns to include
        
        Returns:
            An awaitable that resolves to a pl.DataFrame
        """
        ...

    def selected_chunks_debug(
        self,
        predicate: pl.Expr
    ) -> SelectedChunksDebugReturn: ...
    
    def schema(self, variables: list[str] | None = None) -> Any:
        """Get the schema for the zarr dataset.
        
        Args:
            variables: Optional list of variable names to include
        """
        ...
    
    def root(self) -> str:
        """Get the store root path."""
        ...
    
    def clear_coord_cache(self) -> Any:
        """Clear the coordinate cache (async)."""
        ...
    
    def clear_all_caches(self) -> Any:
        """Clear all caches - metadata and coordinates (async)."""
        ...
    
    def cache_stats(self) -> Any:
        """Get cache statistics (async).
        
        Returns:
            An awaitable that resolves to a dict with:
            - coord_entries: Number of cached coordinate chunks
            - has_metadata: Whether metadata is cached
        """
        ...
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








# @##########################
# TODO: remove
################################

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
