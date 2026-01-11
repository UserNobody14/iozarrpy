# Rainbear

Python + Rust experiment for **lazy Zarr scanning into Polars**, with an API inspired by xarray’s coordinate-based selection.

This repo currently contains:
- A first-pass `scan_zarr(...)` that streams a Zarr store using Rust [`zarrs`] and yields Polars `LazyFrame`s.
- A test suite that compares rainbear against xarray for various Zarr datasets and filter conditions.

## Status / caveats

- **Zarr v3**: the Rust backend uses `zarrs`.
Note that zarrs-v2 is not likely to work for this reason.
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
uv run --with polars python -c "import rainbear; print(rainbear.hello_from_bin())"
```

## Using `scan_zarr`

```python
import polars as pl
import rainbear

lf = rainbear.scan_zarr("/path/to/data.zarr")

# Filter the LazyFrame (predicate's are pushed down and used for chunk pruning)
lf = lf.filter((pl.col("lat") >= 32.0) & (pl.col("lat") <= 52.0))

df = lf.collect()
print(df)
```

## Running the smoke tests

The Python tests create some local Zarr stores and then scan them.

From the workspace root:
```bash
cd rainbear-tests
uv run pytest
```

## Roadmap


### Near Term
- [ ] Geospatial support via ewkb and polars-st
- [ ] Interpolation support
- [ ] Tests against cloud storage backends
- [ ] Benchmarks
- [ ] Documentation

### Longer Term
- [ ] Improved manner of application to take full advantage of Polars' lazy engine
- [ ] Caching Support?
- [ ] Writing to zarr?
- [ ] Capability to work with datatrees
- [ ] Allow output to arrow/pandas/etc.
- [ ] Icechunk support
- [ ] Zarr V2 support (backwards compatibility)



## Code map

- **Rust extension module**: `rainbear/src/lib.rs` exports `_core`
- **Zarr store opener (multi-backend URLs)**: `rainbear/src/zarr_store.rs`
- **Metadata loader (dims/coords/vars + schema)**: `rainbear/src/zarr_meta.rs`
- **Streaming IO source**: `rainbear/src/zarr_source.rs` (exposed to Python as `ZarrSource`)
- **Python API**: `rainbear/src/rainbear/__init__.py` (`scan_random`, `scan_zarr`, `LazyZarrFrame`)
- **Tests**: `rainbear-tests/tests/` (separate workspace package)

[`zarrs`]: https://docs.rs/zarrs/latest/zarrs/