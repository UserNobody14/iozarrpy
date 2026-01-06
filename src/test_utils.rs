use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::{data_type, ArrayBuilder};
use zarrs::group::GroupBuilder;

#[pyfunction]
pub fn _create_demo_store(path: String) -> PyResult<()> {
    let store_path: PathBuf = path.into();
    let store = Arc::new(
        zarrs::filesystem::FilesystemStore::new(&store_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
    );

    // Root group
    let root = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    root.store_metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // time coord (int64)
    let time = ArrayBuilder::new(vec![4], vec![4], data_type::int64(), 0i64)
        .attributes(
            serde_json::json!({"_ARRAY_DIMENSIONS": ["time"]})
                .as_object()
                .unwrap()
                .clone(),
        )
        .build(store.clone(), "/time")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    time.store_metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    time.store_array_subset(&time.subset_all(), &[0i64, 1, 2, 3])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // lat coord (float64)
    let lat = ArrayBuilder::new(vec![3], vec![3], data_type::float64(), f64::NAN)
        .attributes(
            serde_json::json!({"_ARRAY_DIMENSIONS": ["lat"]})
                .as_object()
                .unwrap()
                .clone(),
        )
        .build(store.clone(), "/lat")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    lat.store_metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    lat.store_array_subset(&lat.subset_all(), &[10.0f64, 20.0, 30.0])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // temp var (float64) with dims ["time", "lat"]
    let temp = ArrayBuilder::new(vec![4, 3], vec![2, 3], data_type::float64(), f64::NAN)
        .attributes(
            serde_json::json!({"_ARRAY_DIMENSIONS": ["time", "lat"]})
                .as_object()
                .unwrap()
                .clone(),
        )
        .build(store.clone(), "/temp")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    temp.store_metadata()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    temp.store_array_subset(
        &temp.subset_all(),
        &[
            0.0, 1.0, 2.0, //
            10.0, 11.0, 12.0, //
            20.0, 21.0, 22.0, //
            30.0, 31.0, 32.0, //
        ],
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(())
}

