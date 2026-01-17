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


def create_comprehensive_test_dataset(*, use_cartopy: bool = False):
    """
    Create a 3D dataset with prime-factored chunk grid (7x5x3=105 chunks).
    
    This dataset is designed for rigorous chunk selection testing where:
    - Dimension a: 70 elements, chunk size 10 -> 7 chunks
    - Dimension b: 50 elements, chunk size 10 -> 5 chunks
    - Dimension c: 30 elements, chunk size 10 -> 3 chunks
    
    Coordinate values are designed to map precisely to chunks:
    - a coords: 0..69 (chunk i contains a values i*10..(i+1)*10-1)
    - b coords: 0..49
    - c coords: 0..29
    
    This enables precise arithmetic verification:
    - Selecting a == 15 hits only chunk a=1 -> 1*5*3 = 15 chunks
    - Selecting a < 20 hits chunks a=0,1 -> 2*5*3 = 30 chunks
    - Selecting (a < 20) & (b < 20) -> 2*2*3 = 12 chunks
    - Selecting (a < 20) & (b < 20) & (c < 20) -> 2*2*2 = 8 chunks
    
    Args:
        use_cartopy: If True, use cartopy for coordinate generation (requires cartopy).
                     If False, use simple linear coordinates.
    """
    na, nb, nc = 70, 50, 30
    
    # Create coordinate arrays - these are the values used in predicates
    a_coords = np.arange(na, dtype=np.int64)
    b_coords = np.arange(nb, dtype=np.int64)
    c_coords = np.arange(nc, dtype=np.int64)
    
    # Create data array with deterministic values for verification
    # Value at (i, j, k) = i * 10000 + j * 100 + k
    data = np.zeros((na, nb, nc), dtype=np.float64)
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                data[i, j, k] = i * 10000 + j * 100 + k
    
    # Generate 2D coordinate grids for a subset (for testing 2D coord handling)
    # Note: get_lon_lat_grids returns (len(y), len(x)) shaped arrays
    if use_cartopy and ccrs is not None:
        # Use cartopy to generate realistic lon/lat
        # get_lon_lat_grids expects (x, y) and returns (y, x) shaped arrays
        x_for_grid = np.arange(nc)  # c dimension acts as x
        y_for_grid = np.arange(nb)  # b dimension acts as y
        lon_2d, lat_2d = get_lon_lat_grids(x_for_grid, y_for_grid)
        # Result is (nb, nc) shaped which matches (["b", "c"])
    else:
        # Simple linear fallback - create (nb, nc) shaped arrays directly
        # meshgrid with indexing='ij' gives (nb, nc) shape for (b, c) iteration
        bb, cc = np.meshgrid(np.arange(nb), np.arange(nc), indexing='ij')
        lon_2d = -120.0 + cc.astype(np.float64) * 0.1
        lat_2d = 30.0 + bb.astype(np.float64) * 0.1
    
    ds = xr.Dataset(
        data_vars={
            "data": (["a", "b", "c"], data),
            # Secondary variable for multi-variable tests
            "data2": (["a", "b", "c"], data * 2),
            # 2D variable for testing non-dimension coordinates
            "surface": (["b", "c"], np.random.randn(nb, nc)),
        },
        coords={
            "a": a_coords,
            "b": b_coords,
            "c": c_coords,
            # 2D coordinates for testing multi-dimensional coord handling
            # Shape: (nb, nc) = (50, 30)
            "lon": (["b", "c"], lon_2d),
            "lat": (["b", "c"], lat_2d),
        },
        attrs={
            "description": "Comprehensive test dataset with prime-factored chunk grid",
            "chunk_grid": "7x5x3=105 chunks",
            "coordinate_mode": "cartopy" if (use_cartopy and ccrs is not None) else "fallback",
        },
    )
    return ds


def create_comprehensive_4d_test_dataset():
    """
    Create a 4D dataset for testing time/lead_time style dimensions.
    
    Chunk grid: 3x5x7x4 = 420 chunks (all primes or semi-primes)
    - time: 6 elements, chunk size 2 -> 3 chunks
    - lead: 10 elements, chunk size 2 -> 5 chunks
    - y: 70 elements, chunk size 10 -> 7 chunks
    - x: 40 elements, chunk size 10 -> 4 chunks
    """
    nt, nl, ny, nx = 6, 10, 70, 40
    
    # Time coordinates - datetime
    time_values = [datetime(2024, 1, 1) + timedelta(hours=i * 6) for i in range(nt)]
    # Lead time - duration
    lead_time_values = [timedelta(hours=i) for i in range(nl)]
    
    y_coords = np.arange(ny, dtype=np.int64)
    x_coords = np.arange(nx, dtype=np.int64)
    
    # Deterministic data
    data = np.zeros((nt, nl, ny, nx), dtype=np.float64)
    for t in range(nt):
        for l in range(nl):
            for y in range(ny):
                for x in range(nx):
                    data[t, l, y, x] = t * 1000000 + l * 10000 + y * 100 + x
    
    ds = xr.Dataset(
        data_vars={
            "temperature": (["time", "lead_time", "y", "x"], data),
            "precipitation": (["time", "lead_time", "y", "x"], data * 0.01),
        },
        coords={
            "time": time_values,
            "lead_time": lead_time_values,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "description": "4D comprehensive test dataset",
            "chunk_grid": "3x5x7x4=420 chunks",
        },
    )
    return ds


def create_multi_var_test_dataset():
    """
    Create a dataset with multiple variables for testing variable inference.
    
    This dataset is designed for testing:
    - Variable inference from expressions
    - Per-variable chunk selection
    - Variables with different dimensions
    - Aggregation and window function tests
    
    Variables:
    - temp: 3D (a, b, c) - primary data variable
    - precip: 3D (a, b, c) - secondary data variable
    - wind_u: 3D (a, b, c) - wind component u
    - wind_v: 3D (a, b, c) - wind component v
    - surface: 2D (b, c) - surface-only variable (different dims)
    - pressure: 3D (a, b, c) - another 3D variable
    
    Chunk grid for 3D vars: 5x4x3 = 60 chunks
    - Dimension a: 50 elements, chunk size 10 -> 5 chunks
    - Dimension b: 40 elements, chunk size 10 -> 4 chunks
    - Dimension c: 30 elements, chunk size 10 -> 3 chunks
    
    Chunk grid for 2D var (surface): 4x3 = 12 chunks
    """
    na, nb, nc = 50, 40, 30
    
    # Create coordinate arrays
    a_coords = np.arange(na, dtype=np.int64)
    b_coords = np.arange(nb, dtype=np.int64)
    c_coords = np.arange(nc, dtype=np.int64)
    
    # Create data arrays with distinct patterns for each variable
    # temp: linear in a
    temp = np.zeros((na, nb, nc), dtype=np.float64)
    for i in range(na):
        temp[i, :, :] = 273.15 + i * 0.5  # Temperature in K
    
    # precip: linear in b
    precip = np.zeros((na, nb, nc), dtype=np.float64)
    for j in range(nb):
        precip[:, j, :] = j * 0.1  # Precipitation in mm
    
    # wind_u: linear in c
    wind_u = np.zeros((na, nb, nc), dtype=np.float64)
    for k in range(nc):
        wind_u[:, :, k] = k * 2.0  # Wind u-component in m/s
    
    # wind_v: linear in a + b
    wind_v = np.zeros((na, nb, nc), dtype=np.float64)
    for i in range(na):
        for j in range(nb):
            wind_v[i, j, :] = (i + j) * 0.5  # Wind v-component in m/s
    
    # pressure: linear in all dims
    pressure = np.zeros((na, nb, nc), dtype=np.float64)
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                pressure[i, j, k] = 1000.0 + i * 0.1 + j * 0.2 + k * 0.3
    
    # surface: 2D variable (only b, c dims)
    surface = np.zeros((nb, nc), dtype=np.float64)
    for j in range(nb):
        for k in range(nc):
            surface[j, k] = 100.0 + j + k * 2
    
    # 2D coordinate grids
    bb, cc = np.meshgrid(np.arange(nb), np.arange(nc), indexing='ij')
    lon_2d = -120.0 + cc.astype(np.float64) * 0.1
    lat_2d = 30.0 + bb.astype(np.float64) * 0.1
    
    ds = xr.Dataset(
        data_vars={
            "temp": (["a", "b", "c"], temp),
            "precip": (["a", "b", "c"], precip),
            "wind_u": (["a", "b", "c"], wind_u),
            "wind_v": (["a", "b", "c"], wind_v),
            "pressure": (["a", "b", "c"], pressure),
            "surface": (["b", "c"], surface),
        },
        coords={
            "a": a_coords,
            "b": b_coords,
            "c": c_coords,
            "lon": (["b", "c"], lon_2d),
            "lat": (["b", "c"], lat_2d),
        },
        attrs={
            "description": "Multi-variable test dataset for variable inference testing",
            "chunk_grid_3d": "5x4x3=60 chunks",
            "chunk_grid_2d": "4x3=12 chunks",
            "variables_3d": ["temp", "precip", "wind_u", "wind_v", "pressure"],
            "variables_2d": ["surface"],
        },
    )
    return ds


if __name__ == "__main__":
    # Set the output directory
    output_dir = "output-datasets"
    write_datasets_to_zarr_v3(output_dir)
    output_v2_dir = "output-datasets-v2"
    write_datasets_to_zarr_v2(output_v2_dir)
    # Create sharded Zarr V3 datasets in the same folder as regular V3 datasets
    write_datasets_to_zarr_v3_sharded(output_dir)
