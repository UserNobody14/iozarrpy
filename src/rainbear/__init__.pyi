from __future__ import annotations

from typing import Any

import polars as pl

from ._core import ZarrSource

__all__: list[str]

def hello_from_bin() -> str: ...

def selected_chunks(
    zarr_url: str,
    predicate: pl.Expr,
    variables: list[str] | None = None,
) -> list[dict[str, Any]]: ...

def scan_zarr(
    zarr_url: str,
    *,
    variables: list[str] | None = None,
) -> pl.LazyFrame: ...

def main() -> None: ...


