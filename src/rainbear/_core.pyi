from __future__ import annotations

from typing import Any

import polars as pl

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

def scan_zarr_async(
    zarr_url: str,
    predicate: pl.Expr,
    variables: list[str] | None = None,
    max_concurrency: int | None = None,
    with_columns: list[str] | None = None,
) -> Any: ...

class ZarrSource:
    def __init__(
        self,
        zarr_url: str,
        batch_size: int | None,
        n_rows: int | None,
        variables: list[str] | None = None,
        max_chunks_to_read: int | None = None,
    ) -> None: ...

    def schema(self) -> Any: ...
    def try_set_predicate(self, predicate: pl.Expr) -> None: ...
    def set_with_columns(self, columns: list[str]) -> None: ...
    def next(self) -> pl.DataFrame | None: ...
