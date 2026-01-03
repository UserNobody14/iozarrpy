from typing import Any, Iterator

import polars as pl
from polars.io.plugins import register_io_source

from iozarrpy._core import (
    RandomSource,
    ZarrSource,
    hello_from_bin,
    new_bernoulli,
    new_uniform,
)

__all__ = [
    "RandomSource",
    "ZarrSource",
    "hello_from_bin",
    "new_bernoulli",
    "new_uniform",
    "scan_random",
    "scan_zarr",
    "LazyZarrFrame",
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


class LazyZarrFrame:
    def __init__(self, lf: pl.LazyFrame):
        self._lf = lf

    def sel(self, expr: pl.Expr) -> "LazyZarrFrame":
        return LazyZarrFrame(self._lf.filter(expr))

    def select(self, *args: Any, **kwargs: Any) -> "LazyZarrFrame":
        return LazyZarrFrame(self._lf.select(*args, **kwargs))

    def collect(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        return self._lf.collect(*args, **kwargs)

    async def collect_async(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        return await self._lf.collect_async(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lf, name)


def scan_zarr(
    zarr_url: str,
    *,
    variables: list[str] | None = None,
    size: int = 100_000,
) -> LazyZarrFrame:
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        new_size = size
        if n_rows is not None and n_rows < size:
            new_size = n_rows

        src = ZarrSource(zarr_url, batch_size, new_size, variables)
        if with_columns is not None:
            src.set_with_columns(with_columns)

        predicate_set = True
        if predicate is not None:
            try:
                src.try_set_predicate(predicate)
            except Exception:
                predicate_set = False

        while (out := src.next()) is not None:
            if not predicate_set and predicate is not None:
                out = out.filter(predicate)
            yield out

    # create src again to compute the schema
    src = ZarrSource(zarr_url, 0, 0, variables)
    lf = register_io_source(io_source=source_generator, schema=src.schema())
    return LazyZarrFrame(lf)


def main() -> None:
    print(hello_from_bin())
