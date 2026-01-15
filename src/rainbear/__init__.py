import os
import sys
from typing import Iterator

import polars as pl
from polars.io.plugins import register_io_source

from rainbear._core import (ZarrSource, print_extension_info, scan_zarr_async,
                            selected_chunks)

__all__ = [
    "ZarrSource",
    "print_extension_info",
    "scan_zarr",
    "scan_zarr_async",
    "selected_chunks",
    "main",
]

def scan_zarr(
    zarr_url: str,
    *,
    variables: list[str] | None = None,
    max_chunks_to_read: int | None = None,
) -> pl.LazyFrame:
    """Scan a Zarr store and return a LazyFrame.
    
    Filters applied to this LazyFrame will be pushed down to the Zarr reader
    when possible, enabling efficient reading of large remote datasets.
    """
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        src = ZarrSource(zarr_url, batch_size, n_rows, variables, max_chunks_to_read)
        if with_columns is not None:
            src.set_with_columns(with_columns)

        # Set predicate for constraint extraction (chunk pruning).
        # The Rust side uses constraints to skip chunks but doesn't apply
        # the full predicate filter - we do that here in Python.
        if predicate is not None and os.environ.get("RAINBEAR_PREDICATE_PUSHDOWN", "1") == "1":
            if os.environ.get("RAINBEAR_DEBUG_PREDICATE") == "1":
                print(
                    f"[rainbear] predicate pushed down: type={type(predicate)!r} repr={predicate!r}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                src.try_set_predicate(predicate)
            except Exception as e:
                print(f"[rainbear] constraint extraction failed: {e}", file=sys.stderr, flush=True)
                raise e

        while (out := src.next()) is not None:
            # Always apply predicate in Python for correctness
            if predicate is not None:
                out = out.filter(predicate)
            yield out

    src = ZarrSource(zarr_url, 0, 0, variables, max_chunks_to_read)
    return register_io_source(io_source=source_generator, schema=src.schema())


def main() -> None:
    print(print_extension_info())

