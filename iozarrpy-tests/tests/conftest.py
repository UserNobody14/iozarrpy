"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from zarr.codecs import BloscCodec, BloscShuffle

OUTPUT_DIR = Path(__file__).resolve().parent / "output-datasets"


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
    # HRRR Grid datasets (4D: time, lead_time, y, x)
    # =========================================================================
    
    # Basic grid dataset with default chunking
    DatasetConfig(
        name="hrrr_grid_default",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(50, 150),
    ),
    
    # Grid with chunked encoding
    DatasetConfig(
        name="hrrr_grid_chunked",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(100, 200),
    ),
    
    # Grid with sharding
    DatasetConfig(
        name="hrrr_grid_sharded",
        variables=["2m_temperature", "total_precipitation"],
        expected_cols=["time", "lead_time", "y", "x", "2m_temperature", "total_precipitation"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(50, 150),
    ),
    
    # Constant gradient dataset (unconsolidated)
    DatasetConfig(
        name="hrrr_grid_constant_unconsolidated",
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
        name="hrrr_wind_zlib_little",
        variables=["wind_zlib_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zlib_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(100, 200),
    ),
    DatasetConfig(
        name="hrrr_wind_zlib_big",
        variables=["wind_zlib_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zlib_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(50, 150),
    ),
    DatasetConfig(
        name="hrrr_wind_blosc_little",
        variables=["wind_blosc_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_blosc_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(200, 300),
    ),
    DatasetConfig(
        name="hrrr_wind_blosc_big",
        variables=["wind_blosc_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_blosc_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(150, 250),
    ),
    DatasetConfig(
        name="hrrr_wind_lz4_little",
        variables=["wind_lz4_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(0, 100),
    ),
    DatasetConfig(
        name="hrrr_wind_lz4_big",
        variables=["wind_lz4_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(50, 100),
    ),
    DatasetConfig(
        name="hrrr_wind_lz4hc_little",
        variables=["wind_lz4hc_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4hc_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(250, 350),
    ),
    DatasetConfig(
        name="hrrr_wind_lz4hc_big",
        variables=["wind_lz4hc_big"],
        expected_cols=["time", "lead_time", "y", "x", "wind_lz4hc_big"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="y",
        filter_range=(300, 390),
    ),
    DatasetConfig(
        name="hrrr_wind_zstd_little",
        variables=["wind_zstd_little"],
        expected_cols=["time", "lead_time", "y", "x", "wind_zstd_little"],
        dims=["time", "lead_time", "y", "x"],
        filter_dim="x",
        filter_range=(0, 50),
    ),
    DatasetConfig(
        name="hrrr_wind_zstd_big",
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
        name="hrrr_multi_var",
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
    # HRRR Grid datasets (4D)
    # =========================================================================
    
    # Default chunking
    ds = zarr_generators.create_hrrr_grid_dataset()
    path = make_path("hrrr_grid_default")
    ds.to_zarr(path, zarr_format=3)
    paths["hrrr_grid_default"] = path
    
    # Also use this dataset for wind variable tests
    for var in zarr_generators.get_list_of_variables():
        # Map the config names to the same dataset path
        config_name = f"hrrr_wind_{var.replace('wind_', '')}"
        paths[config_name] = path
    
    # Multi-var uses the same dataset
    paths["hrrr_multi_var"] = path
    
    # Chunked encoding
    ds = zarr_generators.create_hrrr_grid_dataset()
    path = make_path("hrrr_grid_chunked")
    encoding = {
        var: {"chunks": (1, 2, 100, 100), "compressors": [blosc_zstd]}
        for var in ds.data_vars
    }
    ds.to_zarr(path, zarr_format=3, encoding=encoding)
    paths["hrrr_grid_chunked"] = path
    
    # Sharded
    ds = zarr_generators.create_hrrr_grid_dataset()
    path = make_path("hrrr_grid_sharded")
    encoding = {
        var: {"chunks": (1, 2, 100, 100), "shards": (1, 4, 200, 200), "compressors": [blosc_zstd]}
        for var in ds.data_vars
    }
    ds.to_zarr(path, zarr_format=3, encoding=encoding)
    paths["hrrr_grid_sharded"] = path
    
    # Constant gradient, unconsolidated
    ds = zarr_generators.create_hrrr_grid_dataset_constant()
    path = make_path("hrrr_grid_constant_unconsolidated")
    ds.to_zarr(path, zarr_format=3, consolidated=False)
    paths["hrrr_grid_constant_unconsolidated"] = path

    # =========================================================================
    # Orography datasets (2D) with various chunk/shard configs
    # =========================================================================
    
    ds = zarr_generators.create_hrrr_orography_dataset(nx=20, ny=16, sigma=4.0, seed=1)
    path = make_path("orography_chunked_10x10")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (10, 10), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (10, 10), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (10, 10), "compressors": [blosc_zstd]},
    })
    paths["orography_chunked_10x10"] = path

    ds = zarr_generators.create_hrrr_orography_dataset(nx=18, ny=14, sigma=3.5, seed=2)
    path = make_path("orography_chunked_5x5")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (5, 5), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (5, 5), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (5, 5), "compressors": [blosc_zstd]},
    })
    paths["orography_chunked_5x5"] = path

    ds = zarr_generators.create_hrrr_orography_dataset(nx=16, ny=12, sigma=4.0, seed=4)
    path = make_path("orography_sharded_small")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (4, 4), "shards": (8, 8), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (4, 4), "shards": (8, 8), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (4, 4), "shards": (8, 8), "compressors": [blosc_zstd]},
    })
    paths["orography_sharded_small"] = path

    ds = zarr_generators.create_hrrr_orography_dataset(nx=24, ny=20, sigma=5.0, seed=5)
    path = make_path("orography_sharded_large")
    ds.to_zarr(path, zarr_format=3, encoding={
        "geopotential_height": {"chunks": (5, 6), "shards": (10, 12), "compressors": [blosc_zstd]},
        "latitude": {"chunks": (5, 6), "shards": (10, 12), "compressors": [blosc_zstd]},
        "longitude": {"chunks": (5, 6), "shards": (10, 12), "compressors": [blosc_zstd]},
    })
    paths["orography_sharded_large"] = path

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
