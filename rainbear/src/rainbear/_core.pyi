from __future__ import annotations

from typing import Any

import polars as pl

def hello_from_bin() -> str: ...
def _create_demo_store(path: str) -> None: ...


class PySampler:
    ...


def new_bernoulli(name: str, p: float, seed: int) -> PySampler: ...


def new_uniform(
    name: str,
    low: float,
    high: float,
    dtype: Any,
    seed: int,
) -> PySampler: ...


class RandomSource:
    def __init__(
        self,
        columns: list[PySampler],
        size_hint: int | None,
        n_rows: int | None,
    ) -> None: ...

    def schema(self) -> Any: ...
    def try_set_predicate(self, predicate: pl.Expr) -> None: ...
    def set_with_columns(self, columns: list[str]) -> None: ...
    def next(self) -> pl.DataFrame | None: ...


class ZarrSource:
    def __init__(
        self,
        zarr_url: str,
        batch_size: int | None,
        n_rows: int | None,
        variables: list[str] | None = None,
    ) -> None: ...

    def schema(self) -> Any: ...
    def try_set_predicate(self, predicate: pl.Expr) -> None: ...
    def set_with_columns(self, columns: list[str]) -> None: ...
    def next(self) -> pl.DataFrame | None: ...
