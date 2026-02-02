"""Helpers for creating Icechunk repositories from xarray datasets.

These utilities are used by test fixtures to generate Icechunk test data
from the same xarray datasets used for regular Zarr testing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import xarray as xr
from zarr.codecs import BloscCodec, BloscShuffle


def create_icechunk_repo_from_xarray_sync(
    ds: xr.Dataset,
    path: str | Path,
    encoding: dict[str, Any] | None = None,
) -> str:
    """Write an xarray dataset to an Icechunk repository synchronously.

    Args:
        ds: The xarray Dataset to write.
        path: Filesystem path for the Icechunk repository.
        encoding: Optional Zarr encoding dict for variables.

    Returns:
        The string path to the created repository.
    """
    from icechunk import Repository, local_filesystem_storage

    path = Path(path)

    # Clean up existing path
    if path.exists():
        shutil.rmtree(path)

    # Create storage and repository (these are synchronous in icechunk-python)
    storage = local_filesystem_storage(str(path))
    repo = Repository.create(storage)

    # Get a writable session and its store
    session = repo.writable_session("main")
    store = session.store

    # Write dataset to Zarr format 3
    ds.to_zarr(store, zarr_format=3, encoding=encoding)

    # Commit the changes
    session.commit("Initial commit")

    return str(path)


def get_default_blosc_codec() -> BloscCodec:
    """Get the default Blosc codec used across tests."""
    return BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)


def get_orography_encoding(chunk_size: tuple[int, int] = (10, 10)) -> dict[str, Any]:
    """Get standard encoding for orography datasets."""
    blosc = get_default_blosc_codec()
    return {
        "geopotential_height": {"chunks": chunk_size, "compressors": [blosc]},
        "latitude": {"chunks": chunk_size, "compressors": [blosc]},
        "longitude": {"chunks": chunk_size, "compressors": [blosc]},
    }


def get_comprehensive_3d_encoding(chunk_size: int = 10) -> dict[str, Any]:
    """Get standard encoding for comprehensive 3D datasets."""
    blosc = get_default_blosc_codec()
    return {
        "data": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "data2": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "surface": {"chunks": (chunk_size, chunk_size), "compressors": [blosc]},
    }


def get_multi_var_encoding(chunk_size: int = 10) -> dict[str, Any]:
    """Get standard encoding for multi-variable datasets."""
    blosc = get_default_blosc_codec()
    return {
        "temp": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "precip": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "wind_u": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "wind_v": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "pressure": {"chunks": (chunk_size, chunk_size, chunk_size), "compressors": [blosc]},
        "surface": {"chunks": (chunk_size, chunk_size), "compressors": [blosc]},
    }
