"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from zarr.codecs import BloscCodec, BloscShuffle

OUTPUT_DIR = Path(__file__).resolve().parent / "output-datasets"


@pytest.fixture(scope="session", autouse=True)
def _rainbear_safe_mode() -> None:
    """Keep the test runner alive even if predicate pushdown has native issues.

    Pushdown is an optimization; correctness is still ensured because `rainbear.scan_zarr`
    applies the full predicate in Python.
    """
    os.environ.setdefault("RAINBEAR_PREDICATE_PUSHDOWN", "0")


@pytest.fixture
def output_dir() -> Path:
    """Return the output directory path, creating it if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture
def dataset_path(output_dir: Path):
    """Factory fixture to create a clean dataset path."""

    def _dataset_path(name: str) -> str:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
        return str(path)

    return _dataset_path


# ---------------------------------------------------------------------------
# Baseline comparison dataset configurations
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for a test dataset."""

    name: str
    variables: list[str]
    expected_cols: list[str]
    # Dimensions in this dataset
    dims: list[str] = field(default_factory=lambda: ["y", "x"])
    # Optional filter dimensions for selection tests (must be 1D coords)
    filter_dim: str | None = None
    filter_range: tuple[int, int] | None = None


# All dataset configurations for parameterized baseline tests
BASELINE_DATASET_CONFIGS: list[DatasetConfig] = [
    # =========================================================================
    # Grid datasets (4D: time, lead_time, y, x)
    # =========================================================================
    
    # Basic grid dataset with default chunking
    DatasetConfig(
        name="grid_default",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(50, 150),
    ),
    
    # Grid with chunked encoding
    DatasetConfig(
        name="grid_chunked",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(100, 200),
    ),
    
    # Grid with sharding
    DatasetConfig(
        name="grid_sharded",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(50, 150),
    ),
    
    # Constant gradient dataset (unconsolidated)
    DatasetConfig(
        name="grid_constant_unconsolidated",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(0, 100),
    ),
    
    # =========================================================================
    # Wind variables with different compressors (zlib, blosc, lz4, zstd, etc.)
    # =========================================================================
    
    DatasetConfig(
        name="wind_zlib_little",
        variables=["wind_zlib_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zlib_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(100, 200),
    ),
    DatasetConfig(
        name="wind_zlib_big",
        variables=["wind_zlib_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zlib_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(50, 150),
    ),
    DatasetConfig(
        name="wind_blosc_little",
        variables=["wind_blosc_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_blosc_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(200, 300),
    ),
    DatasetConfig(
        name="wind_blosc_big",
        variables=["wind_blosc_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_blosc_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(150, 250),
    ),
    DatasetConfig(
        name="wind_lz4_little",
        variables=["wind_lz4_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(0, 100),
    ),
    DatasetConfig(
        name="wind_lz4_big",
        variables=["wind_lz4_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(50, 100),
    ),
    DatasetConfig(
        name="wind_lz4hc_little",
        variables=["wind_lz4hc_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4hc_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(250, 350),
    ),
    DatasetConfig(
        name="wind_lz4hc_big",
        variables=["wind_lz4hc_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4hc_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(300, 390),
    ),
    DatasetConfig(
        name="wind_zstd_little",
        variables=["wind_zstd_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zstd_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(0, 50),
    ),
    DatasetConfig(
        name="wind_zstd_big",
        variables=["wind_zstd_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zstd_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(100, 200),
    ),
    
    # =========================================================================
    # Multiple variables at once
    # =========================================================================
    
    DatasetConfig(
        name="multi_var",
        variables=["2m_temperature", "wind_zlib_little", "wind_zstd_big"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "wind_zlib_little", "wind_zstd_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(100, 300),
    ),
    
    # =========================================================================
    # Orography datasets (2D: y, x) - different chunk/shard configurations
    # =========================================================================
    
    DatasetConfig(
        name="orography_chunked_10x10",
        variables=["geopotential_height", "latitude", "longitude"],
        expected_cols=["y", "x", "geopotential_height", "latitude", "longitude"],
        dims=["y", "x"],
        filter_dim="y",
        filter_range=(3, 10),
    ),
    DatasetConfig(
        name="orography_chunked_5x5",
        variables=["geopotential_height", "latitude", "longitude"],
        expected_cols=["y", "x", "geopotential_height", "latitude", "longitude"],
        dims=["y", "x"],
        filter_dim="x",
        filter_range=(2, 8),
    ),
    DatasetConfig(
        name="orography_sharded_small",
        variables=["geopotential_height", "latitude", "longitude"],
        expected_cols=["y", "x", "geopotential_height", "latitude", "longitude"],
        dims=["y", "x"],
        filter_dim="y",
        filter_range=(2, 6),
    ),
    DatasetConfig(
        name="orography_sharded_large",
        variables=["geopotential_height", "latitude", "longitude"],
        expected_cols=["y", "x", "geopotential_height", "latitude", "longitude"],
        dims=["y", "x"],
        filter_dim="x",
        filter_range=(4, 12),
    ),
]


def _generate_baseline_datasets(output_dir: Path) -> dict[str, str]:
    """Generate all baseline test datasets and return a mapping of name -> path."""
    from tests import zarr_generators

    paths: dict[str, str] = {}

    def make_path(name: str) -> str:
        path = output_dir / f"{name}.zarr"
        if path.exists():
            shutil.rmtree(path)
        return str(path)

    blosc_zstd = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)

    # =========================================================================
    # Grid datasets (4D)
    # =========================================================================
    
    # Default chunking
    ds = zarr_generators.create_grid_dataset()
    path = make_path("grid_default")
    ds.to_zarr(path, zarr_format=3)
    paths["grid_default"] = path
    
    # Also use this dataset for wind variable tests
    for var in zarr_generators.get_list_of_variables():
        # Map the config names to the same dataset path
        config_name = f"wind_{var.replace('wind_', '')}"
        paths[config_name] = path
    
    # Multi-var uses the same dataset
    paths["multi_var"] = path
    
    # Chunked encoding
    ds = zarr_generators.create_grid_dataset()
    path = make_path("grid_chunked")
    encoding = {
        var: {"chunks": (1, 2, 100, 100), "compressors": [blosc_zstd]}
        for var in ds.data_vars
    }
    ds.to_zarr(path, zarr_format=3, encoding=encoding)
    paths["grid_chunked"] = path
    
    # Sharded
    ds = zarr_generators.create_grid_dataset()
    path = make_path("grid_sharded")
    encoding = {
        var: {"chunks": (1, 2, 100, 100), "shards": (1, 4, 200, 200), "compressors": [blosc_zstd]}
        for var in ds.data_vars
    }
    ds.to_zarr(path, zarr_format=3, encoding=encoding)
    paths["grid_sharded"] = path
    
    # Constant gradient, unconsolidated
    ds = zarr_generators.create_grid_dataset_constant()
    path = make_path("grid_constant_unconsolidated")
    ds.to_zarr(path, zarr_format=3, consolidated=False)
    paths["grid_constant_unconsolidated"] = path

    # =========================================================================
    # Orography datasets (2D) with various chunk/shard configs
    # =========================================================================
    
    ds = zarr_generators.create_orography_dataset(nx=20, ny=16, sigma=4.0, seed=1)
    path = make_path("orography_chunked_10x10")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (10, 10), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (10, 10), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (10, 10), "compressors": [blosc_zstd]},
    })
    paths["orography_chunked_10x10"] = path

    ds = zarr_generators.create_orography_dataset(nx=18, ny=14, sigma=3.5, seed=2)
    path = make_path("orography_chunked_5x5")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (5, 5), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (5, 5), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (5, 5), "compressors": [blosc_zstd]},
    })
    paths["orography_chunked_5x5"] = path

    ds = zarr_generators.create_orography_dataset(nx=16, ny=12, sigma=4.0, seed=4)
    path = make_path("orography_sharded_small")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (4, 4), "shards": (8, 8), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (4, 4), "shards": (8, 8), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (4, 4), "shards": (8, 8), "compressors": [blosc_zstd]},
    })
    paths["orography_sharded_small"] = path

    ds = zarr_generators.create_orography_dataset(nx=24, ny=20, sigma=5.0, seed=5)
    path = make_path("orography_sharded_large")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (5, 6), "shards": (10, 12), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (5, 6), "shards": (10, 12), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (5, 6), "shards": (10, 12), "compressors": [blosc_zstd]},
    })
    paths["orography_sharded_large"] = path

    # =========================================================================
    # Minimal dataset with index-only dims (no 1D coord arrays for y/x)
    # =========================================================================
    import numpy as np
    import zarr

    path = make_path("index_only_dims")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    arr = root.create_array(
        "var",
        shape=(4, 4),
        chunks=(2, 2),
        dtype="f8",
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    arr[:] = np.arange(16, dtype=np.float64).reshape(4, 4)
    paths["index_only_dims"] = path

    return paths


@pytest.fixture(scope="session")
def baseline_datasets() -> dict[str, str]:
    """Session-scoped fixture that generates all baseline test datasets once."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _generate_baseline_datasets(OUTPUT_DIR)


def get_dataset_config(name: str) -> DatasetConfig:
    """Get a dataset configuration by name."""
    for cfg in BASELINE_DATASET_CONFIGS:
        if cfg.name == name:
            return cfg
    raise ValueError(f"Unknown dataset config: {name}")


# ---------------------------------------------------------------------------
# Comprehensive expression test datasets
# ---------------------------------------------------------------------------


@dataclass
class ComprehensiveDatasetInfo:
    """Metadata about the comprehensive test dataset for assertions."""

    path: str
    # Chunk grid shape (a_chunks, b_chunks, c_chunks)
    chunk_grid: tuple[int, int, int]
    # Total number of chunks
    total_chunks: int
    # Chunk size per dimension
    chunk_size: int
    # Dimension lengths
    dim_lengths: tuple[int, int, int]
    # Coordinate mode used
    coord_mode: str


def _generate_comprehensive_datasets(output_dir: Path) -> dict[str, ComprehensiveDatasetInfo]:
    """Generate comprehensive test datasets for expression testing."""
    from tests import zarr_generators

    paths: dict[str, ComprehensiveDatasetInfo] = {}
    blosc_zstd = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)

    # 3D dataset with prime-factored chunk grid: 7x5x3 = 105 chunks
    # Dimensions: a=70 (7 chunks), b=50 (5 chunks), c=30 (3 chunks)
    chunk_size = 10
    chunk_grid = (7, 5, 3)
    dim_lengths = (70, 50, 30)
    total_chunks = 7 * 5 * 3  # = 105

    # Generate with fallback coordinates
    ds = zarr_generators.create_comprehensive_test_dataset(use_cartopy=False)
    path = output_dir / "comprehensive_3d_fallback.zarr"
    if path.exists():
        shutil.rmtree(path)
    encoding = {
        "data": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "data2": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "surface": {"chunks": (chunk_size, chunk_size), "compressors": [blosc_zstd]},
    }
    ds.to_zarr(str(path), zarr_format=3, encoding=encoding)
    paths["comprehensive_3d_fallback"] = ComprehensiveDatasetInfo(
        path=str(path),
        chunk_grid=chunk_grid,
        total_chunks=total_chunks,
        chunk_size=chunk_size,
        dim_lengths=dim_lengths,
        coord_mode="fallback",
    )

    # Generate with cartopy coordinates (if available)
    ds = zarr_generators.create_comprehensive_test_dataset(use_cartopy=True)
    path = output_dir / "comprehensive_3d_cartopy.zarr"
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(str(path), zarr_format=3, encoding=encoding)
    paths["comprehensive_3d_cartopy"] = ComprehensiveDatasetInfo(
        path=str(path),
        chunk_grid=chunk_grid,
        total_chunks=total_chunks,
        chunk_size=chunk_size,
        dim_lengths=dim_lengths,
        coord_mode=ds.attrs.get("coordinate_mode", "unknown"),
    )

    # 4D dataset: 3x5x7x4 = 420 chunks
    ds = zarr_generators.create_comprehensive_4d_test_dataset()
    path = output_dir / "comprehensive_4d.zarr"
    if path.exists():
        shutil.rmtree(path)
    encoding_4d = {
        "temperature": {"chunks": (2, 2, 10, 10), "compressors": [blosc_zstd]},
        "precipitation": {"chunks": (2, 2, 10, 10), "compressors": [blosc_zstd]},
    }
    ds.to_zarr(str(path), zarr_format=3, encoding=encoding_4d)
    paths["comprehensive_4d"] = ComprehensiveDatasetInfo(
        path=str(path),
        chunk_grid=(3, 5, 7, 4),  # type: ignore[arg-type]
        total_chunks=3 * 5 * 7 * 4,  # = 420
        chunk_size=10,  # Spatial chunk size (time/lead are 2)
        dim_lengths=(6, 10, 70, 40),  # type: ignore[arg-type]
        coord_mode="datetime",
    )

    return paths


@pytest.fixture(scope="session")
def comprehensive_datasets() -> dict[str, ComprehensiveDatasetInfo]:
    """Session-scoped fixture providing comprehensive test datasets.

    Returns a dict with keys:
    - 'comprehensive_3d_fallback': 3D dataset (7x5x3=105 chunks) with simple coords
    - 'comprehensive_3d_cartopy': 3D dataset with cartopy coords (or fallback if unavailable)
    - 'comprehensive_4d': 4D dataset (3x5x7x4=420 chunks) with datetime/duration coords
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _generate_comprehensive_datasets(OUTPUT_DIR)


@pytest.fixture(params=["comprehensive_3d_fallback", "comprehensive_3d_cartopy"])
def comprehensive_3d_dataset(
    comprehensive_datasets: dict[str, ComprehensiveDatasetInfo],
    request: pytest.FixtureRequest,
) -> ComprehensiveDatasetInfo:
    """Parameterized fixture that runs tests against both coord modes."""
    return comprehensive_datasets[str(request.param)]


# ---------------------------------------------------------------------------
# Multi-variable dataset for advanced expression testing
# ---------------------------------------------------------------------------


@dataclass
class MultiVarDatasetInfo:
    """Metadata about the multi-variable test dataset."""

    path: str
    # Chunk grid shape for 3D variables (a_chunks, b_chunks, c_chunks)
    chunk_grid_3d: tuple[int, int, int]
    # Chunk grid shape for 2D variables (b_chunks, c_chunks)
    chunk_grid_2d: tuple[int, int]
    # Total chunks for 3D/2D
    total_chunks_3d: int
    total_chunks_2d: int
    # Chunk size per dimension
    chunk_size: int
    # Dimension lengths
    dim_lengths: tuple[int, int, int]
    # 3D variable names
    vars_3d: list[str]
    # 2D variable names
    vars_2d: list[str]
    # All data variable names
    all_data_vars: list[str]


def _generate_multi_var_dataset(output_dir: Path) -> MultiVarDatasetInfo:
    """Generate multi-variable test dataset for advanced expression testing."""
    from tests import zarr_generators

    blosc_zstd = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)

    ds = zarr_generators.create_multi_var_test_dataset()
    path = output_dir / "multi_var.zarr"
    if path.exists():
        shutil.rmtree(path)

    chunk_size = 10
    encoding = {
        "temp": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "precip": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "wind_u": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "wind_v": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "pressure": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc_zstd]},
        "surface": {"chunks": (chunk_size, chunk_size), "compressors": [blosc_zstd]},
    }
    ds.to_zarr(str(path), zarr_format=3, encoding=encoding)

    return MultiVarDatasetInfo(
        path=str(path),
        chunk_grid_3d=(5, 4, 3),
        chunk_grid_2d=(4, 3),
        total_chunks_3d=5 * 4 * 3,  # = 60
        total_chunks_2d=4 * 3,  # = 12
        chunk_size=chunk_size,
        dim_lengths=(50, 40, 30),
        vars_3d=["temp", "precip", "wind_u", "wind_v", "pressure"],
        vars_2d=["surface"],
        all_data_vars=["temp", "precip", "wind_u", "wind_v", "pressure", "surface"],
    )


@pytest.fixture(scope="session")
def multi_var_dataset() -> MultiVarDatasetInfo:
    """Session-scoped fixture providing multi-variable test dataset.

    Dataset has:
    - 5 3D variables: temp, precip, wind_u, wind_v, pressure
    - 1 2D variable: surface
    - 3D chunk grid: 5x4x3 = 60 chunks
    - 2D chunk grid: 4x3 = 12 chunks
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _generate_multi_var_dataset(OUTPUT_DIR)
