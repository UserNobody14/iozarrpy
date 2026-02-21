
import asyncio
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

type AnyZarrBackend = ZarrBackendSync | ZarrBackend | IcechunkBackend

def scan_zarr(
    store_url_or_backend: StoreInput | AnyZarrBackend,
    *,
    max_chunks_to_read: int | None = None,
    prefix: str | None = None,
) -> pl.LazyFrame:
    """Scan a Zarr store and return a LazyFrame.
    
    Filters applied to this LazyFrame will be pushed down to the Zarr reader
    when possible, enabling efficient reading of large remote datasets.
    
    Args:
        store_url_or_backend: Either a URL string (e.g., "s3://bucket/path.zarr")
            or an ObjectStore instance from `rainbear.store` or `obstore`
            or a caching ZarrBackendSync, ZarrBackend, or IcechunkBackend instance.
        max_chunks_to_read: Optional limit on the number of chunks to read (for debugging/safety).
        prefix: Optional path prefix within the store. Only used when passing an ObjectStore
            instance (not needed for URL strings which include the full path).
    
    Examples:
        # Using a URL string (current behavior)
        lf = rainbear.scan_zarr("s3://bucket/path.zarr")
        
        # Using rainbear's own store (full connection pooling)
        s3 = rainbear.store.S3Store(bucket="my-bucket", region="us-east-1")
        lf = rainbear.scan_zarr(s3, prefix="path.zarr")
        
        # Using external obstore (works, but recreated - no shared pool)
        import obstore
        s3 = obstore.store.S3Store(bucket="my-bucket", region="us-east-1")
        lf = rainbear.scan_zarr(s3, prefix="path.zarr")

        # Using a caching ZarrBackendSync, ZarrBackend, or IcechunkBackend instance
        backend = rainbear.ZarrBackendSync.from_url("s3://bucket/path.zarr")
        lf = rainbear.scan_zarr(backend)
    """
    if isinstance(store_url_or_backend, ZarrBackendSync):
        backend = store_url_or_backend
    elif isinstance(store_url_or_backend, ZarrBackend):
        backend = store_url_or_backend
    elif isinstance(store_url_or_backend, IcechunkBackend):
        backend = store_url_or_backend
    else:
        backend = ZarrBackendSync.from_store(store_url_or_backend, prefix=prefix)
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        if isinstance(backend, ZarrBackendSync):
            yield from backend.scan_zarr_streaming_sync(
                predicate=predicate,
                with_columns=with_columns,
                max_chunks_to_read=max_chunks_to_read,
                n_rows=n_rows,
                batch_size=batch_size,
            )
        elif isinstance(backend, IcechunkBackend):
            yield from backend.scan_zarr_streaming_sync(
                predicate=predicate,
                with_columns=with_columns,
                max_chunks_to_read=max_chunks_to_read,
                n_rows=n_rows,
                batch_size=batch_size,
            )
        else:
            # Build the new predicate:
            new_predicate = pl.col(with_columns) if with_columns else pl.all()
            new_predicate = new_predicate.filter(predicate) if predicate else new_predicate
            # Run blocking async scan
            df = asyncio.run(backend.scan_zarr_async(
                predicate=new_predicate,
                max_chunks_to_read=max_chunks_to_read,
            ))
            yield df
    return register_io_source(io_source=source_generator, schema=backend.schema())




def scan_zarr_async(
    store_url_or_backend: StoreInput | AnyZarrBackend,
    predicate: pl.Expr,
    max_concurrency: int | None = None,
    max_chunks_to_read: int | None = None,
    prefix: str | None = None,
) -> Any:
    """Async scan a Zarr store and return a DataFrame.
    
    Args:
        store_url_or_backend: Either a URL string or an ObjectStore instance
            or a caching ZarrBackendSync, ZarrBackend, or IcechunkBackend instance.
        predicate: Filter expression to apply.
        max_concurrency: Maximum number of concurrent chunk reads.
        prefix: Optional path prefix within the store (for ObjectStore instances).
    """
    if isinstance(store_url_or_backend, ZarrBackendSync):
        backend = store_url_or_backend
    elif isinstance(store_url_or_backend, ZarrBackend):
        backend = store_url_or_backend
    elif isinstance(store_url_or_backend, IcechunkBackend):
        backend = store_url_or_backend
    else:
        backend = ZarrBackend.from_store(store_url_or_backend, prefix=prefix)
    if isinstance(backend, ZarrBackendSync):
        return backend.scan_zarr_sync(
            predicate,
            max_chunks_to_read=max_chunks_to_read,
        )
    else:
        return backend.scan_zarr_async(
            predicate,
            max_concurrency=max_concurrency,
            max_chunks_to_read=max_chunks_to_read,
        )















def main() -> None:
    print(print_extension_info())

