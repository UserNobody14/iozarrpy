from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load() -> ModuleType:
    here = Path(__file__).resolve().parent
    target = here / "zarr-gen-test.py"
    spec = importlib.util.spec_from_file_location("zarr_gen_test_impl", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import generator module at {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load()

get_hrrr_lon_lat_grids = _m.get_hrrr_lon_lat_grids
get_list_of_variables = _m.get_list_of_variables
create_hrrr_grid_dataset = _m.create_hrrr_grid_dataset
create_hrrr_grid_dataset_constant = _m.create_hrrr_grid_dataset_constant
create_hrrr_orography_dataset = _m.create_hrrr_orography_dataset

write_datasets_to_zarr_v2 = _m.write_datasets_to_zarr_v2
write_datasets_to_zarr_v3 = _m.write_datasets_to_zarr_v3
write_datasets_to_zarr_v3_sharded = _m.write_datasets_to_zarr_v3_sharded


