
from typing import Any, Iterator

import polars as pl
from polars.io.plugins import register_io_source

from rainbear._core import (
    IcechunkBackend,
    ZarrBackend,
    ZarrBackendSync,
    exceptions,
    print_extension_info,
    store,
)

# Type alias for store input - can be a URL string or an ObjectStore instance
StoreInput = str | Any  # Any here represents ObjectStore instances from store module or obstore

__all__ = [
    "IcechunkBackend",
    "ZarrBackend",
    "ZarrBackendSync",
    "print_extension_info",
    "scan_zarr",
    "store",
    "exceptions",
    "main",
]

def scan_zarr(
    store_or_url: StoreInput,
    *,
    max_chunks_to_read: int | None = None,
    prefix: str | None = None,
) -> pl.LazyFrame:
    """Scan a Zarr store and return a LazyFrame.
    
    Filters applied to this LazyFrame will be pushed down to the Zarr reader
    when possible, enabling efficient reading of large remote datasets.
    
    Args:
        store_or_url: Either a URL string (e.g., "s3://bucket/path.zarr") or an
            ObjectStore instance from `rainbear.store` or `obstore`.
        variables: Optional list of variable names to read. If None, reads all data variables.
        max_chunks_to_read: Optional limit on the number of chunks to read (for debugging/safety).
        prefix: Optional path prefix within the store. Only used when passing an ObjectStore
            instance (not needed for URL strings which include the full path).
    
    Examples:
        # Using a URL string (current behavior)
        df = rainbear.scan_zarr("s3://bucket/path.zarr")
        
        # Using rainbear's own store (full connection pooling)
        s3 = rainbear.store.S3Store(bucket="my-bucket", region="us-east-1")
        df = rainbear.scan_zarr(s3, prefix="path.zarr")
        
        # Using external obstore (works, but recreated - no shared pool)
        import obstore
        s3 = obstore.store.S3Store(bucket="my-bucket", region="us-east-1")
        df = rainbear.scan_zarr(s3, prefix="path.zarr")
    """
    backend = ZarrBackendSync.from_store(store_or_url, prefix=prefix)
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        lf = backend.scan_zarr_sync(
            predicate=predicate,
            with_columns=with_columns,
            max_chunks_to_read=max_chunks_to_read,
            n_rows=n_rows,
            batch_size=batch_size,
        )
        yield lf
    return register_io_source(io_source=source_generator, schema=backend.schema())




def scan_zarr_async(
    store: StoreInput,
    predicate: pl.Expr,
    max_concurrency: int | None = None,
    max_chunks_to_read: int | None = None,
    prefix: str | None = None,
) -> Any:
    """Async scan a Zarr store and return a DataFrame.
    
    Args:
        store: Either a URL string or an ObjectStore instance.
        predicate: Filter expression to apply.
        max_concurrency: Maximum number of concurrent chunk reads.
        prefix: Optional path prefix within the store (for ObjectStore instances).
    """
    backend = ZarrBackend.from_store(store, prefix=prefix)
    return backend.scan_zarr_async(
        predicate,
        max_concurrency=max_concurrency,
        max_chunks_to_read=max_chunks_to_read,
    )















def main() -> None:
    print(print_extension_info())

