use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::sync::Arc;

use polars::prelude::{
    DataType as PlDataType, Field, Schema,
    TimeUnit,
};
use smallvec::SmallVec;
use zarrs::array::ChunkGrid;

use crate::meta::path::ZarrPath;
use crate::{IStr, IntoIStr};

// =============================================================================
// Unified Hierarchical Metadata Types
// =============================================================================

/// Unified metadata for any zarr store (flat or hierarchical).
/// A flat dataset is simply a tree where `root.children` is empty.
#[derive(Debug)]
pub struct ZarrMeta {
    /// Root node of the hierarchy
    pub root: ZarrNode,
    /// Dimension analysis across the entire tree
    pub dim_analysis: DimensionAnalysis,
}

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

/// Dimension analysis across a tree - tracks how dimensions relate across nodes.
#[derive(Debug, Clone, Default)]
pub struct DimensionAnalysis {
    /// All unique dimensions across the tree, in output order (root dims first)
    pub all_dims: Vec<IStr>,
    /// Dimensions from the root node
    pub root_dims: Vec<IStr>,
    /// For each node path, its dimension set
    pub node_dims: BTreeMap<IStr, Vec<IStr>>,
    /// Dimension name -> length
    pub dim_lengths: BTreeMap<IStr, u64>,
}
impl DimensionAnalysis {
    /// Compute output dimension order: root dims first, then extras by first appearance.
    /// Recursively collects dimensions from all nodes in the tree.
    pub fn compute(root: &ZarrNode) -> Self {
        let mut all_dims: Vec<IStr> = Vec::new();
        let mut node_dims: BTreeMap<
            IStr,
            Vec<IStr>,
        > = BTreeMap::new();
        let mut dim_lengths: BTreeMap<IStr, u64> =
            BTreeMap::new();

        // Collect root dims first (they define primary order)
        let root_dims = root.local_dims.clone();
        for dim in &root_dims {
            if !all_dims.contains(dim) {
                all_dims.push(dim.clone());
            }
        }

        // Recursively collect from all nodes
        Self::collect_node(
            root,
            &mut all_dims,
            &mut node_dims,
            &mut dim_lengths,
        );

        Self {
            all_dims,
            root_dims,
            node_dims,
            dim_lengths,
        }
    }

    fn collect_node(
        node: &ZarrNode,
        all_dims: &mut Vec<IStr>,
        node_dims: &mut BTreeMap<IStr, Vec<IStr>>,
        dim_lengths: &mut BTreeMap<IStr, u64>,
    ) {
        // Record this node's dimensions
        node_dims.insert(
            node.path.clone(),
            node.local_dims.clone(),
        );

        // Add any new dimensions not yet seen
        for dim in &node.local_dims {
            if !all_dims.contains(dim) {
                all_dims.push(dim.clone());
            }
        }

        // Infer dim lengths from array shapes
        for (_, arr) in &node.arrays {
            for (i, dim) in
                arr.dims.iter().enumerate()
            {
                if i < arr.shape.len() {
                    dim_lengths
                        .entry(dim.clone())
                        .or_insert(arr.shape[i]);
                }
            }
        }

        // Recurse into children
        for (_, child) in &node.children {
            Self::collect_node(
                child,
                all_dims,
                node_dims,
                dim_lengths,
            );
        }
    }
}

impl ZarrMeta {
    /// True if there are any child groups
    pub fn is_hierarchical(&self) -> bool {
        !self.root.children.is_empty()
    }

    pub fn array_by_path<T: IntoIStr>(
        &self,
        path: T,
    ) -> Option<&ZarrArrayMeta> {
        let zp = ZarrPath::from(path.istr());
        self.root.get_array_recursive(&zp)
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
        self.root
            .collect_data_var_paths_recursive(
                &ZarrPath::root(),
                &mut paths,
            );
        paths
            .into_iter()
            .map(|p| p.to_istr())
            .collect()
    }

    /// All array paths (data vars) using recursive tree traversal.
    pub fn all_array_paths(&self) -> Vec<IStr> {
        self.all_data_var_paths()
    }

    pub fn all_zarr_array_paths(
        &self,
    ) -> Vec<IStr> {
        let mut paths = Vec::new();
        self.root.collect_paths_recursive(
            &ZarrPath::root(),
            &mut paths,
        );
        paths
            .into_iter()
            .map(|p| p.to_istr())
            .collect()
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

        for dim in &self.dim_analysis.all_dims {
            let dtype = self
                .array_by_path(dim)
                .map(|m| m.polars_dtype.clone())
                .unwrap_or(PlDataType::Int64);
            let dim_str: &str = dim.as_ref();
            fields.push(Field::new(
                dim_str.into(),
                dtype,
            ));
        }

        // Add root data variable columns
        for var in &self.root.data_vars {
            let var_str: &str = var.as_ref();
            if var_set
                .as_ref()
                .map_or(true, |vs| {
                    vs.contains(var_str)
                })
            {
                if let Some(m) =
                    self.root.arrays.get(var)
                {
                    fields.push(Field::new(
                        var_str.into(),
                        m.polars_dtype.clone(),
                    ));
                }
            }
        }

        // Add child group struct columns
        for (child_name, child_node) in
            &self.root.children
        {
            let child_name_str: &str =
                child_name.as_ref();
            // Check if this group or any of its vars are in the selection
            let should_include =
                var_set.as_ref().map_or(true, |vs| {
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
                    arr_meta.polars_dtype.clone(),
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
impl Display for ZarrMeta {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "ZarrMeta(root='{}')",
            self.root
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
            out.push(prefix.push(var.clone()));
        }
        for (child_name, child_node) in
            &self.children
        {
            let child_prefix =
                prefix.push(child_name.clone());
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
            out.push(prefix.push(var.clone()));
        }
        for (child_name, child_node) in
            &self.children
        {
            let child_prefix =
                prefix.push(child_name.clone());
            child_node
                .collect_data_var_paths_recursive(
                    &child_prefix,
                    out,
                );
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
                    &ZarrPath::single(
                        comps[0].clone(),
                    ),
                    &mut out,
                );
                out
            }
            Some(child) => {
                child.find_paths_under(
                    &target.tail(),
                )
            }
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
    /// Shape wrapped in Arc for cheap cloning.
    pub shape: Arc<[u64]>,
    /// Regular chunk shape (edge chunks may be smaller).
    pub chunk_shape: Arc<[u64]>,
    pub chunk_grid: Arc<ChunkGrid>,
    pub dims: SmallVec<[IStr; 4]>,
    pub polars_dtype: PlDataType,
    pub encoding: Option<VarEncoding>,
    /// Raw zarrs ArrayMetadata from traverse (for unconsolidated stores)
    pub array_metadata:
        Option<Arc<zarrs::array::ArrayMetadata>>,
}

impl ZarrArrayMeta {
    pub fn chunking_at_dim(
        &self,
        dim: &IStr,
    ) -> Option<u64> {
        let dim_idx = self
            .dims
            .iter()
            .position(|d| d == dim)?;
        if dim_idx >= self.chunk_shape.len() {
            None
        } else {
            Some(self.chunk_shape[dim_idx])
        }
    }
}
