# iozarrpy

Python + Rust experiment for **lazy Zarr scanning into Polars**, with an API inspired by xarray’s coordinate-based selection.

This repo currently contains:
- A working Polars IO plugin pattern (see `scan_random`)
- A first-pass `scan_zarr(...)` that streams a Zarr store using Rust [`zarrs`] and yields Polars `DataFrame` batches
- A thin `LazyZarrFrame` wrapper that provides `.sel(...)` + `.collect_async()`

## Status / caveats

- **Zarr v3**: the Rust backend uses `zarrs` (Zarr v3 + some v2 compatibility).
- **Tidy table output**: `scan_zarr` currently emits a “tidy” `DataFrame` with one row per element and columns:
  - dimension/coord columns (e.g. `time`, `lat`)
  - variable columns (e.g. `temp`)
- **Predicate pushdown**:
  - Rust attempts to compile a limited subset of predicates (simple comparisons on coord columns combined with `&`) for **chunk pruning**.
  - If Polars `Expr` deserialization fails (typically because Python Polars and the Rust-side Polars ABI/serde don’t match), `scan_zarr` automatically falls back to **Python-side filtering** (correct but slower).

## Quickstart (uv)

The project is configured as a `maturin` extension module.

- Run a quick import check:

```bash
uv run --with polars python -c "import iozarrpy; print(iozarrpy.hello_from_bin())"
```

## Using `scan_zarr`

```python
import polars as pl
import iozarrpy

lf = iozarrpy.scan_zarr("/path/to/data.zarr", size=1_000_000)

# xarray-ish selection sugar (currently just a thin wrapper around LazyFrame.filter)
lf = lf.sel((pl.col("lat") >= 32.0) & (pl.col("lat") <= 52.0))

df = lf.collect()
print(df)
```

## Running the smoke tests

The Python tests create a tiny Zarr store via a Rust helper (`iozarrpy._core._create_demo_store`) and then scan it.

From the workspace root:
```bash
cd iozarrpy-tests
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Or use the test script:
```bash
./iozarrpy/scripts/run_tests.sh
```

## Code map

- **Rust extension module**: `iozarrpy/src/lib.rs` exports `_core`
- **Zarr store opener (multi-backend URLs)**: `iozarrpy/src/zarr_store.rs`
- **Metadata loader (dims/coords/vars + schema)**: `iozarrpy/src/zarr_meta.rs`
- **Streaming IO source**: `iozarrpy/src/zarr_source.rs` (exposed to Python as `ZarrSource`)
- **Python API**: `iozarrpy/src/iozarrpy/__init__.py` (`scan_random`, `scan_zarr`, `LazyZarrFrame`)
- **Test fixture helper**: `iozarrpy/src/test_utils.rs` (`_create_demo_store`)
- **Tests**: `iozarrpy-tests/tests/` (separate workspace package)

[`zarrs`]: https://docs.rs/zarrs/latest/zarrs/