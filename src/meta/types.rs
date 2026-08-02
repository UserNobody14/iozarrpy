use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::sync::Arc;

use polars::prelude::{
    DataType as PlDataType, Field, PlSmallStr,
    Schema, TimeUnit,
};
use smallvec::SmallVec;
use zarrs::array::{
    ArrayMetadata, ChunkGrid,
    DataType as ZarrDataType,
};

use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::path::ZarrPath;
use crate::meta::time_encoding::extract_var_encoding_from_attributes;
use crate::shared::{
    IStr, IntoIStr, IntoManyIstrs,
};

// =============================================================================
// Unified Hierarchical Metadata Types
// =============================================================================

/// A node in the zarr hierarchy (group or root).
#[derive(Debug)]
pub struct ZarrNode {
    /// Path from store root (e.g., "/" or "/model_a" or "/level_1/level_2")
    pub path: IStr,
    /// Arrays directly in this node (keyed by leaf name, not full path)
    pub arrays:
        BTreeMap<IStr, Arc<ZarrArrayMeta>>,
    /// Child groups (keyed by child name)
    pub children: BTreeMap<IStr, ZarrNode>,
    /// Dimensions used by arrays in this node
    pub local_dims: Vec<IStr>,
    /// Data variable names (non-coordinate arrays) in this node
    pub data_vars: Vec<IStr>,
}

pub type ZarrMeta = ZarrNode;

impl ZarrNode {
    pub fn array_by_path<T: IntoIStr>(
        &self,
        path: T,
    ) -> Option<&ZarrArrayMeta> {
        let zp = ZarrPath::from(path.istr());
        self.get_array_recursive(&zp)
    }

    pub fn array_by_path_contains<T: IntoIStr>(
        &self,
        path: T,
    ) -> bool {
        self.array_by_path(path).is_some()
    }

    /// All data variable paths using recursive tree traversal.
    pub fn all_data_var_paths(
        &self,
    ) -> Vec<IStr> {
        let mut paths = Vec::new();
        self.collect_data_var_paths_recursive(
            &ZarrPath::root(),
            &mut paths,
        );
        paths.into_istrs()
    }

    /// All coordinate array paths (self-coords and CF aux coords).
    ///
    /// Paths are returned in canonical (no leading slash) flat-string form,
    /// matching `ZarrPath::to_flat_string`. Use [`Self::is_coordinate_path`]
    /// for individual lookups; this is intended for building a hot-path set.
    pub fn all_coord_paths(&self) -> Vec<IStr> {
        let mut paths = Vec::new();
        self.collect_coord_paths_recursive(
            &ZarrPath::root(),
            &mut paths,
        );
        paths.into_istrs()
    }

    /// True if `path` resolves to a coordinate array (not a data variable).
    ///
    /// Returns `false` for unknown paths and for paths that resolve to a
    /// data variable. Accepts paths with or without a leading slash.
    #[allow(dead_code)]
    pub fn is_coordinate_path<T: IntoIStr>(
        &self,
        path: T,
    ) -> bool {
        let zp = ZarrPath::from(path.istr());
        self.is_coord_at(&zp)
    }

    /// All array paths (data vars) using recursive tree traversal.
    pub fn all_array_paths(&self) -> Vec<IStr> {
        self.all_zarr_array_paths()
    }

    pub fn all_zarr_array_paths(
        &self,
    ) -> Vec<IStr> {
        let mut paths = Vec::new();
        self.collect_paths_recursive(
            &ZarrPath::root(),
            &mut paths,
        );
        paths.into_istrs()
    }

    /// Generate a Polars schema for the tidy DataFrame output.
    ///
    /// For flat datasets, this is the same as ZarrDatasetMeta::tidy_schema.
    /// For hierarchical datasets, child groups become struct columns.
    pub fn tidy_schema(
        &self,
        variables: Option<&[IStr]>,
    ) -> Schema {
        let var_set: Option<BTreeSet<&str>> =
            variables.map(|v| {
                v.iter()
                    .map(|s| s.as_ref())
                    .collect()
            });

        let mut fields: Vec<Field> = Vec::new();

        for dim in self.dim_order() {
            let dtype = self
                .array_by_path(dim)
                .map(|m| m.polars_dtype())
                .unwrap_or(PlDataType::Int64);
            let dim_str: &str = dim.as_ref();
            fields.push(Field::new(
                dim_str.into(),
                dtype,
            ));
        }

        let mut field_names: BTreeSet<
            PlSmallStr,
        > = fields
            .iter()
            .map(|f| f.name().clone())
            .collect();

        // Add root data variable columns
        for var in &self.data_vars {
            let var_str: &str = var.as_ref();
            if var_set.as_ref().is_none_or(|vs| {
                vs.contains(var_str)
            }) && let Some(m) =
                self.arrays.get(var)
            {
                fields.push(Field::new(
                    var_str.into(),
                    m.polars_dtype(),
                ));
                field_names
                    .insert(var_str.into());
            }
        }

        // CF-style auxiliary coordinates (lat/lon, …): stored in `arrays` but
        // omitted from `data_vars` when listed as `coordinates` on another var.
        for (var, meta) in &self.arrays {
            let var_str: &str = var.as_ref();
            if field_names.contains(var_str) {
                continue;
            }
            if var_set.as_ref().is_none_or(|vs| {
                vs.contains(var_str)
            }) {
                fields.push(Field::new(
                    var_str.into(),
                    meta.polars_dtype(),
                ));
                field_names
                    .insert(var_str.into());
            }
        }

        // Add child group struct columns
        for (child_name, child_node) in
            &self.children
        {
            let child_name_str: &str =
                child_name.as_ref();
            // Check if this group or any of its vars are in the selection
            let should_include =
                var_set.as_ref().is_none_or(|vs| {
                    vs.contains(child_name_str)
                        || child_node.data_vars.iter().any(
                            |v| {
                                let v_str: &str =
                                    v.as_ref();
                                vs.contains(v_str)
                                    || vs.contains(
                                        &format!(
                                            "{}/{}",
                                            child_name_str,
                                            v_str
                                        )
                                        .as_str(),
                                    )
                            },
                        )
                });

            if should_include {
                let struct_dtype =
                    child_node.to_struct_dtype();
                fields.push(Field::new(
                    child_name_str.into(),
                    struct_dtype,
                ));
            }
        }

        fields.into_iter().collect()
    }

    /// Column names in the same order as [`Self::tidy_schema`].
    pub fn tidy_column_order(
        &self,
        variables: Option<&[IStr]>,
    ) -> Vec<IStr> {
        let var_set: Option<BTreeSet<&str>> =
            variables.map(|v| {
                v.iter()
                    .map(|s| s.as_ref())
                    .collect()
            });

        let mut out: Vec<IStr> = Vec::new();
        let mut seen: BTreeSet<IStr> =
            BTreeSet::new();

        for dim in self.dim_order() {
            out.push(dim);
            seen.insert(dim);
        }

        for var in &self.data_vars {
            let var_str: &str = var.as_ref();
            if var_set.as_ref().is_none_or(|vs| {
                vs.contains(var_str)
            }) && self.arrays.contains_key(var)
            {
                out.push(*var);
                seen.insert(*var);
            }
        }

        for var in self.arrays.keys() {
            let var_str: &str = var.as_ref();
            if seen.contains(var) {
                continue;
            }
            if var_set.as_ref().is_none_or(|vs| {
                vs.contains(var_str)
            }) {
                out.push(*var);
                seen.insert(*var);
            }
        }

        for (child_name, child_node) in
            &self.children
        {
            let child_name_str: &str =
                child_name.as_ref();
            let should_include =
                var_set.as_ref().is_none_or(|vs| {
                    vs.contains(child_name_str)
                        || child_node.data_vars.iter().any(
                            |v| {
                                let v_str: &str =
                                    v.as_ref();
                                vs.contains(v_str)
                                    || vs.contains(
                                        &format!(
                                            "{}/{}",
                                            child_name_str,
                                            v_str
                                        )
                                        .as_str(),
                                    )
                            },
                        )
                });

            if should_include {
                out.push(*child_name);
            }
        }

        out
    }
}

impl ZarrNode {
    /// Create a new empty node at the given path
    pub fn new(path: IStr) -> Self {
        Self {
            path,
            arrays: BTreeMap::new(),
            children: BTreeMap::new(),
            local_dims: Vec::new(),
            data_vars: Vec::new(),
        }
    }

    /// Dimensions across this subtree in output order: this node's
    /// `local_dims` first, then descendants in DFS pre-order.
    pub fn dim_order(&self) -> Vec<IStr> {
        let mut out = Vec::new();
        self.collect_dims(&mut out);
        out
    }

    fn collect_dims(&self, out: &mut Vec<IStr>) {
        for d in &self.local_dims {
            if !out.contains(d) {
                out.push(*d);
            }
        }
        for child in self.children.values() {
            child.collect_dims(out);
        }
    }

    /// True if `name` is a dimension anywhere in this subtree.
    pub fn is_dim(&self, name: &IStr) -> bool {
        self.local_dims.contains(name)
            || self
                .children
                .values()
                .any(|child| child.is_dim(name))
    }

    /// Length of `dim`, from the first array in DFS pre-order that declares it.
    pub fn dim_len(
        &self,
        dim: &IStr,
    ) -> Option<u64> {
        for arr in self.arrays.values() {
            if let Some(i) = arr
                .dims()
                .iter()
                .position(|d| d == dim)
                && let Some(len) =
                    arr.shape().get(i)
            {
                return Some(*len);
            }
        }
        self.children
            .values()
            .find_map(|child| child.dim_len(dim))
    }

    fn to_struct_dtype(&self) -> PlDataType {
        let mut struct_fields: Vec<Field> =
            Vec::new();

        // Add data variable fields
        for var in &self.data_vars {
            if let Some(arr_meta) =
                self.arrays.get(var)
            {
                let var_str: &str = var.as_ref();
                struct_fields.push(Field::new(
                    var_str.into(),
                    arr_meta.polars_dtype(),
                ));
            }
        }

        // Recursively add nested child groups
        for (child_name, child_node) in
            &self.children
        {
            let child_name_str: &str =
                child_name.as_ref();
            let nested_dtype =
                child_node.to_struct_dtype();
            struct_fields.push(Field::new(
                child_name_str.into(),
                nested_dtype,
            ));
        }

        PlDataType::Struct(struct_fields)
    }
}

impl Display for ZarrNode {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "ZarrNode(path='{}')",
            self.path
        )
    }
}

impl ZarrNode {
    /// Look up an array by traversing the tree recursively using path components.
    pub fn get_array_recursive(
        &self,
        path: &ZarrPath,
    ) -> Option<&ZarrArrayMeta> {
        let comps = path.components();
        match comps.len() {
            0 => None,
            1 => self
                .arrays
                .get(&comps[0])
                .map(|a| a.as_ref()),
            _ => self
                .children
                .get(&comps[0])
                .and_then(|child| {
                    child.get_array_recursive(
                        &path.tail(),
                    )
                }),
        }
    }

    /// Recursively collect all array paths (as `ZarrPath`) from this node and descendants.
    pub fn collect_paths_recursive(
        &self,
        prefix: &ZarrPath,
        out: &mut Vec<ZarrPath>,
    ) {
        for var in self.arrays.keys() {
            out.push(prefix.push(*var));
        }
        for (child_name, child_node) in
            &self.children
        {
            let child_prefix =
                prefix.push(*child_name);
            child_node.collect_paths_recursive(
                &child_prefix,
                out,
            );
        }
    }

    /// Recursively collect data variable paths (non-coordinate arrays).
    pub fn collect_data_var_paths_recursive(
        &self,
        prefix: &ZarrPath,
        out: &mut Vec<ZarrPath>,
    ) {
        for var in &self.data_vars {
            out.push(prefix.push(*var));
        }
        for (child_name, child_node) in
            &self.children
        {
            let child_prefix =
                prefix.push(*child_name);
            child_node
                .collect_data_var_paths_recursive(
                    &child_prefix,
                    out,
                );
        }
    }

    /// Recursively collect coordinate array paths.
    ///
    /// Coordinates are arrays in the node that are not listed as data
    /// variables (i.e. self-coords like `lat`/`lon`/`time` and CF aux
    /// coords listed in another array's `coordinates` attribute).
    pub fn collect_coord_paths_recursive(
        &self,
        prefix: &ZarrPath,
        out: &mut Vec<ZarrPath>,
    ) {
        let data_var_set: BTreeSet<&IStr> =
            self.data_vars.iter().collect();
        for name in self.arrays.keys() {
            if !data_var_set.contains(name) {
                out.push(prefix.push(*name));
            }
        }
        for (child_name, child_node) in
            &self.children
        {
            let child_prefix =
                prefix.push(*child_name);
            child_node
                .collect_coord_paths_recursive(
                    &child_prefix,
                    out,
                );
        }
    }

    /// True if `path` resolves to an array under this node that is a
    /// coordinate (i.e. exists in `arrays` but not in `data_vars`).
    #[allow(dead_code)]
    pub fn is_coord_at(
        &self,
        path: &ZarrPath,
    ) -> bool {
        let comps = path.components();
        match comps.len() {
            0 => false,
            1 => {
                self.arrays
                    .contains_key(&comps[0])
                    && !self
                        .data_vars
                        .contains(&comps[0])
            }
            _ => self
                .children
                .get(&comps[0])
                .is_some_and(|child| {
                    child
                        .is_coord_at(&path.tail())
                }),
        }
    }

    /// Find all paths under this node that match a given prefix path.
    /// Traverses to the target node then collects all paths beneath it.
    pub fn find_paths_under(
        &self,
        target: &ZarrPath,
    ) -> Vec<ZarrPath> {
        let comps = target.components();
        if comps.is_empty() {
            let mut out = Vec::new();
            self.collect_paths_recursive(
                &ZarrPath::root(),
                &mut out,
            );
            return out;
        }
        match self.children.get(&comps[0]) {
            Some(child) if comps.len() == 1 => {
                let mut out = Vec::new();
                child.collect_paths_recursive(
                    &ZarrPath::single(comps[0]),
                    &mut out,
                );
                out
            }
            Some(child) => child
                .find_paths_under(&target.tail()),
            None => Vec::new(),
        }
    }
}

/// CF-conventions time encoding information parsed from Zarr attributes.
#[derive(Debug, Clone)]
pub struct TimeEncoding {
    /// The epoch (reference timestamp) in nanoseconds since Unix epoch.
    pub epoch_ns: i64,
    /// Multiplier to convert stored units to nanoseconds.
    pub unit_ns: i64,
    /// Whether this is a duration (timedelta) rather than a datetime.
    pub is_duration: bool,
}

impl TimeEncoding {
    #[inline]
    pub fn decode(&self, raw: i64) -> i64 {
        if self.is_duration {
            raw.saturating_mul(self.unit_ns)
        } else {
            raw.saturating_mul(self.unit_ns)
                .saturating_add(self.epoch_ns)
        }
    }

    /// Decode a float value (e.g. CF "days since epoch" stored as float64)
    /// to nanoseconds. Used when coordinate arrays are stored as float.
    #[inline]
    pub fn decode_f64(
        &self,
        raw: f64,
    ) -> Option<i64> {
        if !raw.is_finite() {
            return None;
        }
        let unit_ns_f = self.unit_ns as f64;
        let scaled = raw * unit_ns_f;
        let ns = scaled.clamp(
            i64::MIN as f64,
            i64::MAX as f64,
        ) as i64;
        let ns = if self.is_duration {
            ns
        } else {
            ns.saturating_add(self.epoch_ns)
        };
        Some(ns)
    }

    pub fn to_polars_dtype(&self) -> PlDataType {
        if self.is_duration {
            PlDataType::Duration(
                TimeUnit::Nanoseconds,
            )
        } else {
            PlDataType::Datetime(
                TimeUnit::Nanoseconds,
                None,
            )
        }
    }
}

/// Unified encoding for CF-convention variable transformations.
///
/// Covers both time encoding (units since epoch) and scale/offset
/// packing (e.g. satellite data stored as int16 with scale_factor
/// and add_offset attributes).
#[derive(Debug, Clone)]
pub enum VarEncoding {
    /// CF time encoding: raw integer values represent
    /// time units since an epoch. Decoded as:
    /// `raw * unit_ns + epoch_ns`, then cast to
    /// Datetime or Duration.
    Time(TimeEncoding),
    /// CF scale/offset packing: raw packed values
    /// (typically int16) represent floating-point data.
    /// Decoded as: `raw * scale_factor + add_offset`.
    ScaleOffset {
        scale_factor: f64,
        add_offset: f64,
        /// Raw fill value (in packed space); matching
        /// elements become NaN after decoding.
        fill_value: Option<f64>,
    },
}

impl VarEncoding {
    /// The Polars dtype that decoded output should use.
    pub fn decoded_polars_dtype(
        &self,
    ) -> PlDataType {
        match self {
            VarEncoding::Time(te) => {
                te.to_polars_dtype()
            }
            VarEncoding::ScaleOffset {
                ..
            } => PlDataType::Float64,
        }
    }

    /// Extract the inner `TimeEncoding` if this is the `Time` variant.
    pub fn as_time_encoding(
        &self,
    ) -> Option<&TimeEncoding> {
        match self {
            VarEncoding::Time(te) => Some(te),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct ZarrArrayMeta {
    pub path: IStr,
    outer_chunk_grid: Arc<ChunkGrid>,
    inner_chunk_grid: Option<Arc<ChunkGrid>>,
    /// Raw zarrs metadata from traversal.
    metadata: Arc<ArrayMetadata>,
}

impl ZarrArrayMeta {
    pub fn new(
        path: IStr,
        outer_chunk_grid: Arc<ChunkGrid>,
        inner_chunk_grid: Option<Arc<ChunkGrid>>,
        metadata: Arc<ArrayMetadata>,
    ) -> Self {
        Self {
            path,
            outer_chunk_grid,
            inner_chunk_grid,
            metadata,
        }
    }

    pub fn metadata_arc(
        &self,
    ) -> Arc<ArrayMetadata> {
        Arc::clone(&self.metadata)
    }

    #[allow(dead_code)]
    pub fn raw_shape(&self) -> &[u64] {
        match &*self.metadata {
            ArrayMetadata::V2(metadata) => {
                &metadata.shape
            }
            ArrayMetadata::V3(metadata) => {
                &metadata.shape
            }
        }
    }

    #[allow(dead_code)]
    pub fn attributes(
        &self,
    ) -> &serde_json::Map<String, serde_json::Value>
    {
        match &*self.metadata {
            ArrayMetadata::V2(metadata) => {
                &metadata.attributes
            }
            ArrayMetadata::V3(metadata) => {
                &metadata.attributes
            }
        }
    }

    pub fn shape(&self) -> &[u64] {
        self.outer_chunk_grid.array_shape()
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn is_1d(&self) -> bool {
        self.ndim() == 1
    }

    pub fn read_chunk_shape(&self) -> Box<[u64]> {
        let zeros = vec![0u64; self.ndim()];
        self.read_chunk_grid()
            .chunk_shape_u64(&zeros)
            .ok()
            .flatten()
            .map(|cs| {
                cs.to_vec().into_boxed_slice()
            })
            .unwrap_or_else(|| {
                self.raw_shape()
                    .to_vec()
                    .into_boxed_slice()
            })
    }

    pub fn dims(&self) -> SmallVec<[IStr; 4]> {
        if let Some(v) = self
            .attributes()
            .get("_ARRAY_DIMENSIONS")
            && let Some(list) = v.as_array()
        {
            let out: SmallVec<[IStr; 4]> = list
                .iter()
                .filter_map(|x| {
                    x.as_str().map(|s| s.istr())
                })
                .collect();
            if !out.is_empty() {
                return out;
            }
        }

        if let ArrayMetadata::V3(metadata) =
            &*self.metadata
            && let Some(names) =
                &metadata.dimension_names
        {
            return names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    n.as_ref()
                        .map(|s| {
                            s.as_str().istr()
                        })
                        .unwrap_or_else(|| {
                            format!("dim_{i}")
                                .istr()
                        })
                })
                .collect();
        }

        (0..self.raw_shape().len())
            .map(|i| format!("dim_{i}").istr())
            .collect()
    }

    pub fn zarr_dtype(&self) -> ZarrDataType {
        match &*self.metadata {
            ArrayMetadata::V2(metadata) => {
                ZarrDataType::from_metadata(
                    &metadata.dtype,
                )
            }
            ArrayMetadata::V3(metadata) => {
                ZarrDataType::from_metadata(
                    &metadata.data_type,
                )
            }
        }
        .expect("array metadata was validated by zarrs during traversal")
    }

    pub fn polars_dtype(&self) -> PlDataType {
        zarr_dtype_to_polars(
            &self.zarr_dtype(),
            self.encoding().as_ref(),
        )
    }

    pub fn encoding(
        &self,
    ) -> Option<VarEncoding> {
        extract_var_encoding_from_attributes(
            self.attributes(),
        )
    }

    pub fn outer_chunk_grid(&self) -> &ChunkGrid {
        &self.outer_chunk_grid
    }

    pub fn inner_chunk_grid(
        &self,
    ) -> Option<&ChunkGrid> {
        self.inner_chunk_grid.as_deref()
    }

    pub fn read_chunk_grid(
        &self,
    ) -> Arc<ChunkGrid> {
        self.inner_chunk_grid
            .as_ref()
            .map_or_else(
                || {
                    Arc::clone(
                        &self.outer_chunk_grid,
                    )
                },
                Arc::clone,
            )
    }
}
