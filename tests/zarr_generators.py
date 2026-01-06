import shutil
from datetime import datetime, timedelta
from pathlib import Path

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - optional test dependency
    ccrs = None
import numpy as np
import xarray as xr
from numcodecs import LZ4, Blosc, Zlib, Zstd
from zarr.codecs import BloscCodec, BloscShuffle


def get_lon_lat_grids(x, y):
    """
    Compute the lat/lon grid corresponding to a given set of x/y indices
    centered at the central latitude/longitude.
    """
    # Optional dependency: cartopy is heavy and not needed for most tests.
    # If it's not installed, fall back to a simple synthetic lon/lat grid.
    if ccrs is None:
        xx, yy = np.meshgrid(x, y)
        lon = -130.0 + (xx.astype(np.float64) * 0.02)
        lat = 20.0 + (yy.astype(np.float64) * 0.02)
        return lon, lat

    center_x = x[(len(x) - 1) // 2]
    center_y = y[(len(y) - 1) // 2]
    lambert = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )

    plate = ccrs.PlateCarree()
    grid_size = 3000
    xx, yy = np.meshgrid((x - center_x) * grid_size, (y - center_y) * grid_size)
    transformer = plate.transform_points(lambert, xx, yy)
    lon, lat = transformer[..., 0], transformer[..., 1]

    return lon, lat


def get_list_of_variables():
    """
    Create a series of additional variables to be used with various compressors, sharding, endianness, etc.
    """
    # Create a series of additional variables to be used with various compressors, sharding, endianness, etc.
    list_of_compressors = [
        "zlib",
        "blosc",
        "lz4",
        "lz4hc",
        "blosclz",
        "snappy",
        "zstd",
    ]
    list_of_endianness = [
        "little",
        "big",
    ]
    list_of_variables = [
        f"wind_{compressor}_{endianness}"
        for compressor in list_of_compressors for endianness in list_of_endianness
    ]
    return list_of_variables

def create_grid_dataset():
    """
    Create a synthetic dataset with a 'full' grid of lat/lon coordinates
    """
    # Define grid dimensions
    nx, ny = 400, 400
    nt = 3  # number of time points
    nl = 10  # number of lead times

    # Create a mesh grid
    x = np.arange(nx)
    y = np.arange(ny)
    lon, lat = get_lon_lat_grids(x, y)

    # Create timestamps
    time_values = [datetime(2024, 1, 1) + timedelta(hours=i * 6) for i in range(nt)]
    lead_time_values = [timedelta(hours=i) for i in range(nl)]

    # Create synthetic data arrays with realistic values
    temp_data = 273.15 + 10 * np.random.randn(nt, nl, ny, nx)  # temperatures around 0C
    precip_data = np.maximum(
        0, 0.01 * np.random.exponential(1, size=(nt, nl, ny, nx))
    )  # precipitation in m


    list_of_variables = get_list_of_variables()
    wind_data = {
        variable: np.linspace(0, 1, nt * nl * ny * nx).reshape(nt, nl, ny, nx)
        for variable in list_of_variables
    }
    # Flip byte endianness for big endian variables
    for variable in list_of_variables:
        if variable.endswith("_big"):
            wind_data[variable] = wind_data[variable].byteswap(inplace=True)
    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "2m_temperature": (["time", "lead_time", "y", "x"], temp_data),
            "total_precipitation": (["time", "lead_time", "y", "x"], precip_data),
            **{
                variable: (["time", "lead_time", "y", "x"], wind_data[variable])
                for variable in list_of_variables
            }
        },
        coords={
            "time": time_values,
            "lead_time": lead_time_values,
            "latitude": (["y", "x"], lat),
            "longitude": (["y", "x"], lon),
            "x": x,
            "y": y,
        },
        attrs={"projection": "lambert_conformal"},
    )
    return ds


def create_grid_dataset_constant():
    """
    Create a synthetic dataset with linear gradients for testing interpolation.
    """
    # Define grid dimensions
    nx, ny = 400, 400
    nt = 3  # number of time points
    nl = 10  # number of lead times

    # Create a mesh grid
    x = np.arange(nx)
    y = np.arange(ny)
    lon, lat = get_lon_lat_grids(x, y)

    # Create timestamps
    time_values = [datetime(2024, 1, 1) + timedelta(hours=i * 6) for i in range(nt)]
    lead_time_values = [timedelta(hours=i) for i in range(nl)]

    # Create data arrays with linear gradients
    # Temperature: 273.15K (0°C) at western edge to 313.15K (40°C) at eastern edge
    temp_gradient = np.linspace(273.15, 313.15, nx, endpoint=True)
    temp_2d = np.tile(temp_gradient, (ny, 1))
    temp_data = np.tile(temp_2d, (nt, nl, 1, 1))

    # Precipitation: 0mm at southern edge to 10mm at northern edge
    precip_gradient = np.linspace(0, 0.01, ny, endpoint=True)
    precip_2d = np.tile(precip_gradient[:, np.newaxis], (1, nx))
    precip_data = np.tile(precip_2d, (nt, nl, 1, 1))

    list_of_variables = get_list_of_variables()
    def gen_wind_data(variable):
        wind_gradient = np.linspace(1, 20, ny, endpoint=True)
        wind_2d = np.tile(wind_gradient[:, np.newaxis], (1, nx))
        return np.tile(wind_2d, (nt, nl, 1, 1))
    wind_data = {
        variable: gen_wind_data(variable)
        for variable in list_of_variables
    }
    # # Flip byte endianness for big endian variables
    # for variable in list_of_variables:
    #     if variable.endswith("_big"):
    #         wind_data[variable] = wind_data[variable].byteswap(inplace=True)
    
    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "2m_temperature": (["time", "lead_time", "y", "x"], temp_data),
            "total_precipitation": (["time", "lead_time", "y", "x"], precip_data),
            **{
                variable: (["time", "lead_time", "y", "x"], wind_data[variable])
                for variable in list_of_variables
            }
        },
        coords={
            "time": time_values,
            "lead_time": lead_time_values,
            "latitude": (["y", "x"], lat),
            "longitude": (["y", "x"], lon),
            "x": x,
            "y": y,
        },
        attrs={"projection": "lambert_conformal"},
    )
    return ds


def create_orography_dataset(
    *,
    nx: int = 400,
    ny: int = 400,
    sigma: float = 10.0,
    base_elevation: float = 500.0,
    seed: int = 42,
):
    """
    Create a synthetic orography dataset with a Gaussian hill.
    """
    # Define grid dimensions
    # Base elevation (meters)

    # Create coordinates
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    lon, lat = get_lon_lat_grids(x, y)

    # Create gaussian hill with peak of 2000m
    center_x = (nx - 1) // 2
    center_y = (ny - 1) // 2
    elevation = base_elevation + 2000 * np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)
    )

    # Add some random variation
    np.random.seed(seed)  # For reproducibility
    elevation += np.random.normal(0, 50, size=(ny, nx))

    ds = xr.Dataset(
        data_vars={
            "geopotential_height": (["y", "x"], elevation),
            "latitude": (["y", "x"], lat),
            "longitude": (["y", "x"], lon),
        },
        coords={
            "y": y,
            "x": x,
        },
    )
    return ds


def remove_all_files_in_path(path):
    """
    Remove all files in the given path
    """
    for file in path.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)


def write_datasets_to_zarr_v2(output_dir):
    """
    Create and write the test datasets to Zarr V2 format with chunking and blosc compression
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset 1: Grid dataset with random values
    print("Creating grid dataset...")
    ds1 = create_grid_dataset()
    ds1_path = output_path / "grid_dataset.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds1_path)


    print(f"Writing to {ds1_path}")
    ds1["2m_temperature"].encoding.update(
        compressors=[
            {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": 1,
            }
        ],
    )
    ds1["total_precipitation"].encoding.update(
        compressors=[
           Zlib(level=1)
        ],
    )
    for var in ds1.data_vars:
        # If the variable is in the list of variables (starts with "wind_"), update the encoding and endianness
        if var.startswith("wind_"):
            encoding = var.split("_")[1]
            endianness = var.split("_")[2]
            if encoding == "zlib":
                ds1[var].encoding.update(
                    compressors=[
                        Zlib(level=1)
                    ],
                )
            elif encoding == "blosc":
                ds1[var].encoding.update(
                    compressors=[
                        Blosc(cname="zstd", clevel=5)
                    ],
                )
            elif encoding == "lz4":
                ds1[var].encoding.update(
                    compressors=[
                        LZ4(acceleration=3)
                    ],
                )
            elif encoding == "lz4hc":
                ds1[var].encoding.update(
                    compressors=[
                        Blosc(cname="lz4hc", clevel=3)
                    ],
                )
            # elif encoding == "blosclz":
            #     ds1[var].encoding.update(
            #         compressors=[
            #             Blosc(cname="blosclz", clevel=3)
            #         ],
            #     )
            # elif encoding == "snappy":
            #     ds1[var].encoding.update(
            #         compressors=[
            #             Blosc(cname="snappy", clevel=3)
            #         ],
            #     )
            elif encoding == "zstd":
                ds1[var].encoding.update(
                    compressors=[
                        # Blosc(cname="zstd", clevel=3, shuffle=1)
                        Zstd(level=3)
                    ],
                )
            if endianness == "big":
                ds1[var].encoding.update(
                    dtype=">f8",
                )
            elif endianness == "little":
                ds1[var].encoding.update(
                    dtype="<f8",
                )
    ds1.to_zarr(ds1_path, zarr_format=2)
    print("Done!")


    # Dataset 2: Grid dataset with constant gradient (and not consolidated)
    print("Creating grid dataset with constant gradient...")
    ds2 = create_grid_dataset_constant()
    ds2_path = output_path / "grid_dataset_constant.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds2_path)

    print(f"Writing to {ds2_path}")
    ds2.to_zarr(ds2_path, zarr_format=2, consolidated=False)

    print("Done!")



def write_datasets_to_zarr_v3(output_dir):
    """
    Create and write the test datasets to Zarr V3 format with chunking and blosc compression
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create and save each dataset
    print("Creating datasets...")

    # Dataset 1: Grid dataset with random values
    print("Creating grid dataset...")
    ds1 = create_grid_dataset()
    ds1_path = output_path / "grid_dataset.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds1_path)

    # Define chunking and compression for ds1
    encoding1 = {
        var: {
            "chunks": (1, 2, 100, 100),
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        }
        for var in ds1.data_vars
    }

    print(f"Writing to {ds1_path}")
    ds1.to_zarr(ds1_path, encoding=encoding1, zarr_format=3)
    print("Done!")

    # Dataset 2: Grid dataset with constant gradient
    print("Creating grid dataset with constant gradient...")
    ds2 = create_grid_dataset_constant()
    ds2_path = output_path / "grid_dataset_constant.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds2_path)

    blosc = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
    for var in ds2.data_vars:
        ds2.data_vars[var].encoding.update(
            compressors=[blosc],
            chunks=(1, 2, 100, 100),
        )
        ds2.data_vars[var].attrs["fill_value"] = "NaN"

    print(f"Writing to {ds2_path}")
    ds2.to_zarr(ds2_path, zarr_format=3, consolidated=False)
    print("Done!")

    # Dataset 3: Orography dataset
    print("Creating orography dataset...")
    ds3 = create_orography_dataset()
    ds3_path = output_path / "orography_dataset.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds3_path)

    # Define chunking and compression for ds3
    encoding3 = {
        var: {
            "chunks": (100, 100),
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        }
        if var == "geopotential_height"
        else {
            "chunks": (400, 400),
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        }
        for var in ds3.data_vars
    }

    print(f"Writing to {ds3_path}")
    ds3.to_zarr(ds3_path, encoding=encoding3, zarr_format=3)
    print("Done!")

    print(
        f"All datasets have been written to {output_dir} in Zarr V3 format with chunking and blosc compression."
    )


def write_datasets_to_zarr_v3_sharded(output_dir):
    """
    Create and write test datasets to Zarr V3 format with sharding enabled.
    Based on the discussion at: https://github.com/pydata/xarray/discussions/9938
    
    Key points for sharding:
    - Zarr shards must be evenly divisible by Dask chunks
    - Use 'shards' parameter in encoding alongside 'chunks'
    - Sharding can improve performance for certain access patterns
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating datasets with Zarr V3 sharding...")

    # Dataset 1: Grid dataset with sharding
    print("Creating grid dataset with sharding...")
    ds1 = create_grid_dataset()
    ds1_path = output_path / "grid_dataset_sharded.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds1_path)

    # Define sharding configuration
    # Dask chunks: (1, 2, 100, 100) 
    # Zarr shards: (1, 4, 200, 200) - evenly divisible by Dask chunks
    encoding1 = {
        var: {
            "chunks": (1, 2, 100, 100),  # Dask chunks
            "shards": (1, 4, 200, 200),  # Zarr shards - must be evenly divisible by chunks
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        }
        for var in ds1.data_vars
    }

    print(f"Writing sharded dataset to {ds1_path}")
    print("Shard configuration: (1, 4, 200, 200)")
    print("Chunk configuration: (1, 2, 100, 100)")
    ds1.to_zarr(ds1_path, encoding=encoding1, zarr_format=3)
    print("Done!")

    # Dataset 2: Orography dataset with different sharding pattern
    print("Creating orography dataset with sharding...")
    ds2 = create_orography_dataset()
    ds2_path = output_path / "orography_dataset_sharded.zarr"

    # Delete everything from path
    remove_all_files_in_path(ds2_path)

    # Define sharding for 2D data
    # For geopotential_height: chunks (100, 100), shards (200, 200)
    # For lat/lon coordinates: chunks (200, 200), shards (400, 400)
    encoding2 = {
        "geopotential_height": {
            "chunks": (100, 100),
            "shards": (200, 200),  # 2x2 chunks per shard
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        },
        "latitude": {
            "chunks": (200, 200),
            "shards": (400, 400),  # 2x2 chunks per shard
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        },
        "longitude": {
            "chunks": (200, 200),
            "shards": (400, 400),  # 2x2 chunks per shard
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
            ],
        },
    }

    print(f"Writing sharded orography dataset to {ds2_path}")
    print("Geopotential height - Shards: (200, 200), Chunks: (100, 100)")
    print("Lat/Lon coordinates - Shards: (400, 400), Chunks: (200, 200)")
    ds2.to_zarr(ds2_path, encoding=encoding2, zarr_format=3)
    print("Done!")

    print(
        f"Sharded datasets have been written to {output_dir} in Zarr V3 format with sharding enabled."
    )


if __name__ == "__main__":
    # Set the output directory
    output_dir = "output-datasets"
    write_datasets_to_zarr_v3(output_dir)
    output_v2_dir = "output-datasets-v2"
    write_datasets_to_zarr_v2(output_v2_dir)
    # Create sharded Zarr V3 datasets in the same folder as regular V3 datasets
    write_datasets_to_zarr_v3_sharded(output_dir)
