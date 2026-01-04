from typing import Any, Iterator

import polars as pl
from polars.io.plugins import register_io_source
from rainbear._core import (RandomSource, ZarrSource, hello_from_bin,
                            new_bernoulli, new_uniform)

__all__ = [
    "RandomSource",
    "ZarrSource",
    "hello_from_bin",
    "new_bernoulli",
    "new_uniform",
    "scan_random",
    "scan_zarr",
    "main",
]


def scan_random(samplers: list[Any], size: int = 1000) -> pl.LazyFrame:
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source.
        This function will be registered as IO source.
        """

        new_size = size
        if n_rows is not None and n_rows < size:
            new_size = n_rows

        src = RandomSource(samplers, batch_size, new_size)
        if with_columns is not None:
            src.set_with_columns(with_columns)

        # Set the predicate.
        predicate_set = True
        if predicate is not None:
            try:
                src.try_set_predicate(predicate)
            except Exception:
                predicate_set = False

        while (out := src.next()) is not None:
            # If the source could not apply the predicate
            # (because it wasn't able to deserialize it), we do it here.
            if not predicate_set and predicate is not None:
                out = out.filter(predicate)

            yield out

    # create src again to compute the schema
    src = RandomSource(samplers, 0, 0)
    return register_io_source(io_source=source_generator, schema=src.schema())


def scan_zarr(
    zarr_url: str,
    *,
    variables: list[str] | None = None,
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
        src = ZarrSource(zarr_url, batch_size, n_rows, variables)
        if with_columns is not None:
            src.set_with_columns(with_columns)

        # Set predicate for constraint extraction (chunk pruning).
        # The Rust side uses constraints to skip chunks but doesn't apply
        # the full predicate filter - we do that here in Python.
        if predicate is not None:
            try:
                src.try_set_predicate(predicate)
            except Exception:
                pass  # Constraint extraction failed, no chunk pruning

        while (out := src.next()) is not None:
            # Always apply predicate in Python for correctness
            if predicate is not None:
                out = out.filter(predicate)
            yield out

    src = ZarrSource(zarr_url, 0, 0, variables)
    return register_io_source(io_source=source_generator, schema=src.schema())


def main() -> None:
    print(hello_from_bin())


def main() -> None:
    print(hello_from_bin())
