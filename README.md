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
uv run --with polars python -c "import rainbear; print(rainbear.print_extension_info())"
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

## Caching Backends

Rainbear provides three backend classes that own the store connection and cache metadata and coordinate chunks across multiple scans. This dramatically improves performance for repeated queries on the same dataset.

### `ZarrBackend` (Async)

The **async caching backend** for standard Zarr stores. Best for cloud storage (S3, GCS, Azure) where async I/O provides significant performance benefits.

**Features:**
- Persistent caching of coordinate array chunks and metadata across scans
- Async I/O with configurable concurrency for parallel chunk reads
- Compatible with any ObjectStore (S3, GCS, Azure, HTTP, local filesystem)
- Cache statistics and management (clear cache, view stats)

**When to use:**
- Cloud-based Zarr stores where network latency dominates
- Applications already using async/await patterns
- High-concurrency workloads with many simultaneous chunk reads

```python
import polars as pl
from datetime import datetime
import rainbear

# Create backend from URL
backend = rainbear.ZarrBackend.from_url("s3://bucket/dataset.zarr")

# First scan - reads and caches coordinates
df1 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 1, 1))

# Second scan - reuses cached coordinates (much faster!)
df2 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 6, 1))

# Check what's cached
stats = await backend.cache_stats()
print(f"Cached {stats['coord_entries']} coordinate chunks")

# Clear cache if needed
await backend.clear_coord_cache()
```

### `ZarrBackendSync` (Sync)

The **synchronous caching backend** for standard Zarr stores. Best for local filesystem access or simpler synchronous codebases.

**Features:**
- Same persistent caching as `ZarrBackend` (coordinates and metadata)
- Synchronous API - no async/await required
- Blocking I/O suitable for local or low-latency stores
- Additional options: column selection, row limits, batch size control

**When to use:**
- Local filesystem Zarr stores
- Synchronous applications or scripts
- Interactive data exploration (notebooks, REPL)
- When you don't need async concurrency

```python
import polars as pl
from datetime import datetime
import rainbear

# Create backend from URL
backend = rainbear.ZarrBackendSync.from_url("/path/to/local/dataset.zarr")

# Scan with column selection and row limit
df1 = backend.scan_zarr_sync(
    predicate=pl.col("time") > datetime(2024, 1, 1),
    with_columns=["temp", "pressure"],
    n_rows=1000
)

# Second scan reuses cached coordinates
df2 = backend.scan_zarr_sync(pl.col("time") > datetime(2024, 6, 1))

# No await needed for cache operations in sync backend
stats = backend.cache_stats()
backend.clear_coord_cache()
```

### `IcechunkBackend` (Async, Version Control)

The **async-only caching backend** for [Icechunk](https://icechunk.io/)-backed Zarr stores. Icechunk adds Git-like version control to Zarr datasets, enabling branches, commits, and time-travel queries.

**Features:**
- Same persistent caching as `ZarrBackend` (coordinates and metadata)
- Access to versioned Zarr data with branch/snapshot support
- Direct integration with icechunk-python Session objects
- Async-only (Icechunk operations are inherently async)

**When to use:**
- Working with version-controlled Zarr datasets
- Need to query specific branches or historical snapshots
- Collaborative workflows with multiple dataset versions
- Reproducible analysis requiring exact dataset versions

```python
import polars as pl
from datetime import datetime
import rainbear

# Create backend from Icechunk filesystem repository
backend = await rainbear.IcechunkBackend.from_filesystem(
    path="/path/to/icechunk/repo",
    branch="main"  # or specific branch name
)

# Scan like normal - caching works the same
df1 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 1, 1))
df2 = await backend.scan_zarr_async(pl.col("time") > datetime(2024, 6, 1))

# Or use existing Icechunk session directly
from icechunk import Repository, local_filesystem_storage

storage = local_filesystem_storage("/path/to/repo")
repo = Repository.open(storage)
session = repo.readonly_session("experimental-branch")

# No manual serialization needed!
backend = await rainbear.IcechunkBackend.from_session(session)
df = await backend.scan_zarr_async(pl.col("lat") < 45.0)
```

### Backend Comparison

| Feature | ZarrBackend | ZarrBackendSync | IcechunkBackend |
|---------|-------------|-----------------|-----------------|
| **API Style** | Async | Sync | Async |
| **Caching** | ✓ Coordinates & metadata | ✓ Coordinates & metadata | ✓ Coordinates & metadata |
| **Best For** | Cloud storage (S3, GCS, Azure) | Local filesystem | Version-controlled datasets |
| **Concurrency** | High (configurable) | Single-threaded | High (configurable) |
| **Version Control** | ✗ | ✗ | ✓ (branches, snapshots) |
| **Column Selection** | ✗ | ✓ | ✗ |
| **Row Limits** | ✗ | ✓ | ✗ |

## Running the smoke tests

The Python tests create some local Zarr stores and then scan them.

From the workspace root:
```bash
cd rainbear-tests
uv run pytest
```

# Development

To run the Rust tests:
```bash
cargo test
```

To run the Python tests:
```bash
uv run pytest
```

Profiling:
```bash
samply record -- uv run python -m pytest tests/test_benchmark_novel_queries.py -m 'benchmark' --no-header -rN
```


## Roadmap


### Near Term
- [ ] Geospatial support via ewkb and polars-st
- [x] Interpolation support
- [ ] Tests against cloud storage backends
- [x] Benchmarks
- [ ] Documentation

### Longer Term
- [ ] Improved manner of application to take full advantage of Polars' lazy engine
- [x] Caching Support?
- [ ] Writing to zarr?
- [x] Capability to work with datatrees
- [ ] Allow output to arrow/pandas/etc.
- [x] Icechunk support
- [ ] Zarr V2 support (backwards compatibility)



## Code map

- **Rust extension module**: `rainbear/src/lib.rs` exports `_core`
- **Zarr store opener (multi-backend URLs)**: `rainbear/src/zarr_store.rs`
- **Metadata loader (dims/coords/vars + schema)**: `rainbear/src/zarr_meta.rs`
- **Streaming IO source**: `rainbear/src/zarr_source.rs` (exposed to Python as `ZarrBackendSync`)
- **Python API**: `rainbear/src/rainbear/__init__.py` (`scan_random`, `scan_zarr`, `ZarrBackendSync`)
- **Tests**: `rainbear-tests/tests/` (separate workspace package)

[`zarrs`]: https://docs.rs/zarrs/latest/zarrs/