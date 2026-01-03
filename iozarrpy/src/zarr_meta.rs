use std::collections::{BTreeMap, BTreeSet};

use polars::prelude::{DataType as PlDataType, Field, Schema};
use zarrs::array::Array;
use zarrs::hierarchy::NodeMetadata;

use crate::zarr_store::{open_store, OpenedStore};

#[derive(Debug, Clone)]
pub struct ZarrArrayMeta {
    pub name: String,
    pub path: String,
    pub shape: Vec<u64>,
    pub dims: Vec<String>,
    pub zarr_dtype: String,
    pub polars_dtype: PlDataType,
}

#[derive(Debug, Clone)]
pub struct ZarrDatasetMeta {
    pub root: String,
    /// Arrays indexed by their resolved column name (currently leaf name, de-duped if needed).
    pub arrays: BTreeMap<String, ZarrArrayMeta>,
    pub dims: Vec<String>,
    pub coords: Vec<String>,
    pub data_vars: Vec<String>,
}

impl ZarrDatasetMeta {
    /// Build a “tidy table” schema: coord columns + variable columns.
    ///
    /// If a dimension has a matching 1D coordinate array (same name), the coord column uses that dtype.
    /// Otherwise the coord column is an `Int64` index.
    pub fn tidy_schema(&self, variables: Option<&[String]>) -> Schema {
        let var_set: Option<BTreeSet<&str>> = variables.map(|v| v.iter().map(|s| s.as_str()).collect());

        let mut fields: Vec<Field> = Vec::new();

        for dim in &self.dims {
            let dtype = self
                .arrays
                .get(dim)
                .map(|m| m.polars_dtype.clone())
                .unwrap_or(PlDataType::Int64);
            fields.push(Field::new(dim.into(), dtype));
        }

        let vars_iter: Box<dyn Iterator<Item = &str>> = if let Some(var_set) = &var_set {
            Box::new(self.data_vars.iter().map(|s| s.as_str()).filter(|v| var_set.contains(v)))
        } else {
            Box::new(self.data_vars.iter().map(|s| s.as_str()))
        };

        for v in vars_iter {
            if let Some(m) = self.arrays.get(v) {
                fields.push(Field::new(v.into(), m.polars_dtype.clone()));
            }
        }

        fields.into_iter().collect()
    }
}

pub fn load_dataset_meta(zarr_url: &str) -> Result<ZarrDatasetMeta, String> {
    let OpenedStore { store, root } = open_store(zarr_url)?;
    load_dataset_meta_from_opened(&OpenedStore { store, root })
}

pub fn load_dataset_meta_from_opened(opened: &OpenedStore) -> Result<ZarrDatasetMeta, String> {
    let store = opened.store.clone();
    let root = opened.root.clone();

    // Open the root group and traverse nodes.
    let group = zarrs::group::Group::open(store.clone(), &root).map_err(to_string_err)?;
    let nodes = group.traverse().map_err(to_string_err)?;

    let mut arrays: BTreeMap<String, ZarrArrayMeta> = BTreeMap::new();
    let mut seen_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut dims_union: BTreeSet<String> = BTreeSet::new();
    let mut coord_candidates: BTreeMap<String, (Vec<u64>, PlDataType)> = BTreeMap::new();

    for (path, md) in nodes {
        if !matches!(md, NodeMetadata::Array(_)) {
            continue;
        }

        let path_str = path.as_str().to_string();
        let leaf = leaf_name(&path_str);

        let array = Array::open(store.clone(), &path_str).map_err(to_string_err)?;

        let shape = array.shape().to_vec();
        let dims = dims_for_array(&array).unwrap_or_else(|| default_dims(shape.len()));
        for d in &dims {
            dims_union.insert(d.clone());
        }

        // Candidate coord: 1D array whose leaf name equals its (only) dim name.
        if shape.len() == 1 && dims.len() == 1 && leaf == dims[0] {
            let dt = zarr_dtype_to_polars(array.data_type().identifier());
            coord_candidates.insert(leaf.clone(), (shape.clone(), dt));
        }

        let zarr_dtype = array.data_type().identifier().to_string();
        let polars_dtype = zarr_dtype_to_polars(&zarr_dtype);

        // De-dupe by leaf name if collisions happen (nested groups).
        let name = match seen_names.get_mut(&leaf) {
            None => {
                seen_names.insert(leaf.clone(), 1);
                leaf.clone()
            }
            Some(n) => {
                *n += 1;
                format!("{leaf}__{n}")
            }
        };

        arrays.insert(
            name.clone(),
            ZarrArrayMeta {
                name,
                path: path_str,
                shape,
                dims,
                zarr_dtype,
                polars_dtype,
            },
        );
    }

    // Determine dataset dims (stable order).
    let dims: Vec<String> = dims_union.into_iter().collect();

    // Determine coords: any dim that has a matching 1D coordinate array.
    let mut coords: Vec<String> = Vec::new();
    for dim in &dims {
        if let Some((shape, _dtype)) = coord_candidates.get(dim) {
            // Must match dimension length (if dim length is known from any data var).
            // We keep it simple: accept any 1D coord; later we’ll validate against vars.
            if shape.len() == 1 {
                coords.push(dim.clone());
            }
        }
    }

    // Determine data variables: arrays that are not classified as coords by name.
    let coord_set: BTreeSet<&str> = coords.iter().map(|s| s.as_str()).collect();
    let data_vars: Vec<String> = arrays
        .keys()
        .filter(|k| !coord_set.contains(k.as_str()))
        .cloned()
        .collect();

    Ok(ZarrDatasetMeta {
        root,
        arrays,
        dims,
        coords,
        data_vars,
    })
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn leaf_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or_default().to_string()
}

fn default_dims(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("dim_{i}")).collect()
}

fn dims_for_array<TStorage: ?Sized>(array: &Array<TStorage>) -> Option<Vec<String>> {
    // Prefer xarray-style attribute, when present.
    if let Some(v) = array.attributes().get("_ARRAY_DIMENSIONS") {
        if let Some(list) = v.as_array() {
            let out: Vec<String> = list
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect();
            if !out.is_empty() {
                return Some(out);
            }
        }
    }

    // Fall back to Zarr V3 dimension_names.
    if let Some(names) = array.dimension_names() {
        let out: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(i, n)| n.clone().unwrap_or_else(|| format!("dim_{i}")))
            .collect();
        return Some(out);
    }

    None
}

fn zarr_dtype_to_polars(zarr_identifier: &str) -> PlDataType {
    // Conservative, first-milestone mapping.
    match zarr_identifier {
        "bool" => PlDataType::Boolean,
        "int8" => PlDataType::Int8,
        "int16" => PlDataType::Int16,
        "int32" => PlDataType::Int32,
        "int64" => PlDataType::Int64,
        "uint8" => PlDataType::UInt8,
        "uint16" => PlDataType::UInt16,
        "uint32" => PlDataType::UInt32,
        "uint64" => PlDataType::UInt64,
        "float16" | "bfloat16" => PlDataType::Float32,
        "float32" => PlDataType::Float32,
        "float64" => PlDataType::Float64,
        "string" => PlDataType::String,
        // Keep unknowns representable; we can error later in the reader if needed.
        _ => PlDataType::Binary,
    }
}

