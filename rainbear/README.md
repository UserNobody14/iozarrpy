# Rainbear

Build lazy Zarr scans with Polars.

## Installation

```bash
pip install rainbear
```
or with uv

```bash
uv add rainbear
```
## Usage

```python
import rainbear

lf = rainbear.scan_zarr("path/to/zarr")
df = lf.filter(pl.col("x") > 5).collect()
```

