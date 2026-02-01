import os
import sys
from typing import Any, Iterator

import polars as pl
from polars.io.plugins import register_io_source

from rainbear._core import (
    ZarrBackend,
    ZarrSource,
    exceptions,
    print_extension_info,
    store,
)

# Type alias for store input - can be a URL string or an ObjectStore instance
StoreInput = str | Any  # Any here represents ObjectStore instances from store module or obstore

__all__ = [
    "ZarrBackend",
    "ZarrSource",
    "print_extension_info",
    "scan_zarr",
    "store",
    "exceptions",
    "main",
]

def scan_zarr(
    store_or_url: StoreInput,
    *,
    variables: list[str] | None = None,
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
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        src = ZarrSource(store_or_url, batch_size, n_rows, variables, max_chunks_to_read, prefix)
        if with_columns is not None:
            src.set_with_columns(with_columns)

        if os.environ.get("RAINBEAR_DEBUG_PREDICATE") == "1":
            print(
                f"[rainbear] predicate pushed down: type={type(predicate)!r} repr={predicate!r}",
                file=sys.stderr,
                flush=True,
            )
        try:
            if predicate is not None:
                src.try_set_predicate(predicate)
        except Exception as e:
            print(f"[rainbear] constraint extraction failed: {e}", file=sys.stderr, flush=True)
            raise e

        while (out := src.next()) is not None:
            # Always apply predicate in Python for correctness
            if predicate is not None:
                out = out.filter(predicate)
            yield out

    src = ZarrSource(store_or_url, 0, 0, variables, max_chunks_to_read, prefix)
    return register_io_source(io_source=source_generator, schema=src.schema())



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
    backend = ZarrBackend.from_store(store, prefix=prefix)
    return backend.scan_zarr_async(predicate, variables, max_concurrency, with_columns)















def main() -> None:
    print(print_extension_info())

