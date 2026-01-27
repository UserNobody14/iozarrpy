"""Generators for hierarchical zarr stores using xarray DataTree.

This module creates zarr stores with nested group hierarchies for testing
rainbear's handling of DataTree-like structures represented as Polars structs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import xarray as xr


def create_simple_datatree(
    *,
    nx: int = 20,
    ny: int = 16,
    seed: int = 42,
) -> xr.DataTree:
    """Create a simple 2-level DataTree with shared dimensions.

    Structure:
        /
        ├── surface (y, x)        # root-level variable
        ├── model_a/
        │   ├── temperature (y, x)
        │   └── pressure (y, x)
        └── model_b/
            ├── temperature (y, x)
            └── humidity (y, x)

    All nodes share the same (y, x) coordinate system.
    """
    np.random.seed(seed)

    # Shared coordinates
    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)

    # Root dataset with surface variable
    root_ds = xr.Dataset(
        data_vars={
            "surface": (["y", "x"], np.random.randn(ny, nx).astype(np.float64)),
        },
        coords={
            "y": y,
            "x": x,
        },
        attrs={"description": "Root level of simple datatree"},
    )

    # Model A child dataset
    model_a_ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["y", "x"],
                273.15 + 20 * np.random.randn(ny, nx).astype(np.float64),
            ),
            "pressure": (
                ["y", "x"],
                1013.25 + 50 * np.random.randn(ny, nx).astype(np.float64),
            ),
        },
        attrs={"model_name": "model_a", "version": "1.0"},
    )

    # Model B child dataset
    model_b_ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["y", "x"],
                273.15 + 25 * np.random.randn(ny, nx).astype(np.float64),
            ),
            "humidity": (
                ["y", "x"],
                0.5 + 0.3 * np.random.randn(ny, nx).astype(np.float64),
            ),
        },
        attrs={"model_name": "model_b", "version": "2.0"},
    )

    # Build the DataTree
    tree = xr.DataTree.from_dict(
        {
            "/": root_ds,
            "/model_a": model_a_ds,
            "/model_b": model_b_ds,
        }
    )

    return tree


def create_model_ensemble_tree(
    *,
    nx: int = 30,
    ny: int = 24,
    nt: int = 6,
    n_members: int = 5,
    seed: int = 123,
) -> xr.DataTree:
    """Create a realistic ensemble model structure with time dimension.

    Structure:
        /
        ├── time (1D coord)
        ├── y (1D coord)
        ├── x (1D coord)
        ├── orography (y, x)          # static field at root
        ├── member_0/
        │   ├── temperature (time, y, x)
        │   ├── precipitation (time, y, x)
        │   └── wind_speed (time, y, x)
        ├── member_1/
        │   └── ...
        └── member_N/
            └── ...

    This mimics an ensemble forecast structure where each member
    has its own set of forecast variables.
    """
    np.random.seed(seed)

    # Shared coordinates
    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)
    time = np.arange(nt, dtype=np.int64)

    # Root dataset with static orography
    xx, yy = np.meshgrid(x, y)
    center_x, center_y = nx // 2, ny // 2
    orography = 500 + 1500 * np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * 5**2)
    )

    root_ds = xr.Dataset(
        data_vars={
            "orography": (["y", "x"], orography.astype(np.float64)),
        },
        coords={
            "time": time,
            "y": y,
            "x": x,
        },
        attrs={"description": "Ensemble forecast root"},
    )

    # Build tree dict starting with root
    tree_dict: dict[str, xr.Dataset] = {"/": root_ds}

    # Create ensemble members
    for i in range(n_members):
        member_ds = xr.Dataset(
            data_vars={
                "temperature": (
                    ["time", "y", "x"],
                    273.15
                    + 20 * np.random.randn(nt, ny, nx).astype(np.float64)
                    + i * 0.5,  # slight bias per member
                ),
                "precipitation": (
                    ["time", "y", "x"],
                    np.maximum(
                        0, 5 * np.random.exponential(1, size=(nt, ny, nx))
                    ).astype(np.float64),
                ),
                "wind_speed": (
                    ["time", "y", "x"],
                    np.abs(10 + 5 * np.random.randn(nt, ny, nx)).astype(np.float64),
                ),
            },
            attrs={"member_id": i, "description": f"Ensemble member {i}"},
        )
        tree_dict[f"/member_{i}"] = member_ds

    return xr.DataTree.from_dict(tree_dict)


def create_deep_datatree(
    *,
    depth: int = 4,
    nx: int = 16,
    ny: int = 12,
    seed: int = 777,
) -> xr.DataTree:
    """Create a deeply nested DataTree for stress testing.

    Structure (depth=4):
        /
        ├── root_var (y, x)
        ├── level_1/
        │   ├── var_1 (y, x)
        │   └── level_2/
        │       ├── var_2 (y, x)
        │       └── level_3/
        │           ├── var_3 (y, x)
        │           └── level_4/
        │               └── var_4 (y, x)

    Each level adds a nested child with its own variable.
    """
    np.random.seed(seed)

    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)

    # Root dataset
    root_ds = xr.Dataset(
        data_vars={
            "root_var": (["y", "x"], np.random.randn(ny, nx).astype(np.float64)),
        },
        coords={
            "y": y,
            "x": x,
        },
        attrs={"level": 0},
    )

    tree_dict: dict[str, xr.Dataset] = {"/": root_ds}

    # Build nested path
    current_path = ""
    for level in range(1, depth + 1):
        current_path = f"{current_path}/level_{level}"
        level_ds = xr.Dataset(
            data_vars={
                f"var_{level}": (
                    ["y", "x"],
                    (level * 10 + np.random.randn(ny, nx)).astype(np.float64),
                ),
            },
            attrs={"level": level, "path": current_path},
        )
        tree_dict[current_path] = level_ds

    return xr.DataTree.from_dict(tree_dict)


def create_heterogeneous_tree(
    *,
    nx: int = 20,
    ny: int = 16,
    nz: int = 8,
    seed: int = 999,
) -> xr.DataTree:
    """Create a DataTree with heterogeneous schemas (different dims per node).

    Structure:
        /
        ├── y, x (shared coords)
        ├── surface_2d/
        │   ├── elevation (y, x)
        │   └── land_mask (y, x)
        ├── atmosphere_3d/
        │   ├── z (additional coord)
        │   ├── temperature (z, y, x)
        │   └── humidity (z, y, x)
        └── timeseries_1d/
            ├── station_id (new dim)
            └── measurement (station_id)

    This tests handling of nodes with different dimensionalities.
    """
    np.random.seed(seed)

    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)
    z = np.arange(nz, dtype=np.int64)
    station_id = np.arange(10, dtype=np.int64)

    # Root with just coords
    root_ds = xr.Dataset(
        coords={
            "y": y,
            "x": x,
        },
        attrs={"description": "Heterogeneous tree root"},
    )

    # 2D surface node
    surface_ds = xr.Dataset(
        data_vars={
            "elevation": (["y", "x"], 100 * np.random.rand(ny, nx).astype(np.float64)),
            "land_mask": (
                ["y", "x"],
                (np.random.rand(ny, nx) > 0.3).astype(np.float64),
            ),
        },
        attrs={"node_type": "surface_2d"},
    )

    # 3D atmosphere node with additional z dimension
    atmosphere_ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["z", "y", "x"],
                273.15 + np.random.randn(nz, ny, nx).astype(np.float64),
            ),
            "humidity": (
                ["z", "y", "x"],
                np.clip(0.5 + 0.3 * np.random.randn(nz, ny, nx), 0, 1).astype(
                    np.float64
                ),
            ),
        },
        coords={
            "z": z,
        },
        attrs={"node_type": "atmosphere_3d"},
    )

    # 1D timeseries node with completely different dimension
    timeseries_ds = xr.Dataset(
        data_vars={
            "measurement": (
                ["station_id"],
                np.random.randn(len(station_id)).astype(np.float64),
            ),
        },
        coords={
            "station_id": station_id,
        },
        attrs={"node_type": "timeseries_1d"},
    )

    return xr.DataTree.from_dict(
        {
            "/": root_ds,
            "/surface_2d": surface_ds,
            "/atmosphere_3d": atmosphere_ds,
            "/timeseries_1d": timeseries_ds,
        }
    )


def create_wide_datatree(
    *,
    n_children: int = 10,
    nx: int = 12,
    ny: int = 10,
    seed: int = 555,
) -> xr.DataTree:
    """Create a wide DataTree with many sibling children.

    Structure:
        /
        ├── y, x (coords)
        ├── child_0/
        │   └── data (y, x)
        ├── child_1/
        │   └── data (y, x)
        └── ... (n_children total)

    Tests horizontal scaling of struct fields.
    """
    np.random.seed(seed)

    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)

    root_ds = xr.Dataset(
        coords={
            "y": y,
            "x": x,
        },
        attrs={"description": "Wide tree root"},
    )

    tree_dict: dict[str, xr.Dataset] = {"/": root_ds}

    for i in range(n_children):
        child_ds = xr.Dataset(
            data_vars={
                "data": (
                    ["y", "x"],
                    (i * 100 + np.random.randn(ny, nx)).astype(np.float64),
                ),
            },
            attrs={"child_index": i},
        )
        tree_dict[f"/child_{i}"] = child_ds

    return xr.DataTree.from_dict(tree_dict)


def write_datatree_to_zarr(
    tree: xr.DataTree,
    path: str | Path,
    *,
    zarr_format: int = 3,
    consolidated: bool = True,
    chunk_size: int | None = None,
) -> str:
    """Write a DataTree to zarr format.

    Args:
        tree: The DataTree to write.
        path: Output path.
        zarr_format: Zarr format version (2 or 3).
        consolidated: Whether to consolidate metadata.
        chunk_size: Optional chunk size for all dimensions.

    Returns:
        The path as a string.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)

    # Apply chunking encoding if specified
    if chunk_size is not None:
        # Note: DataTree.to_zarr handles encoding per-node
        # For simplicity, we write without explicit encoding and let zarr choose defaults
        pass

    tree.to_zarr(str(path), zarr_format=zarr_format, consolidated=consolidated)
    return str(path)


# Convenience functions for generating test stores


def generate_simple_datatree_store(output_dir: Path, name: str = "simple_datatree.zarr") -> str:
    """Generate a simple datatree zarr store for testing."""
    tree = create_simple_datatree()
    path = output_dir / name
    return write_datatree_to_zarr(tree, path)


def generate_ensemble_tree_store(
    output_dir: Path, name: str = "ensemble_tree.zarr"
) -> str:
    """Generate an ensemble model tree zarr store for testing."""
    tree = create_model_ensemble_tree()
    path = output_dir / name
    return write_datatree_to_zarr(tree, path)


def generate_deep_tree_store(
    output_dir: Path, depth: int = 4, name: str | None = None
) -> str:
    """Generate a deeply nested tree zarr store for testing."""
    tree = create_deep_datatree(depth=depth)
    if name is None:
        name = f"deep_tree_d{depth}.zarr"
    path = output_dir / name
    return write_datatree_to_zarr(tree, path)


def generate_heterogeneous_tree_store(
    output_dir: Path, name: str = "heterogeneous_tree.zarr"
) -> str:
    """Generate a heterogeneous schema tree zarr store for testing."""
    tree = create_heterogeneous_tree()
    path = output_dir / name
    return write_datatree_to_zarr(tree, path)


def generate_wide_tree_store(
    output_dir: Path, n_children: int = 10, name: str | None = None
) -> str:
    """Generate a wide tree zarr store for testing."""
    tree = create_wide_datatree(n_children=n_children)
    if name is None:
        name = f"wide_tree_n{n_children}.zarr"
    path = output_dir / name
    return write_datatree_to_zarr(tree, path)
