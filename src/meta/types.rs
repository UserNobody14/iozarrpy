use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::sync::Arc;

use polars::prelude::{
    DataType as PlDataType, Field, Schema,
    TimeUnit,
};
use smallvec::SmallVec;
use zarrs::array::ChunkGrid;

use crate::{IStr, IntoIStr};

// =============================================================================
// Unified Hierarchical Metadata Types
// =============================================================================

/// Unified metadata for any zarr store (flat or hierarchical).
/// A flat dataset is simply a tree where `root.children` is empty.
#[derive(Debug, Clone)]
pub struct ZarrMeta {
    /// Root node of the hierarchy
    pub root: ZarrNode,
    /// Dimension analysis across the entire tree
    pub dim_analysis: DimensionAnalysis,
    /// Fast lookup: array path (e.g., "model_a/temperature") -> array metadata
    pub path_to_array:
        BTreeMap<IStr, ZarrArrayMeta>,
}

/// A node in the zarr hierarchy (group or root).
#[derive(Debug, Clone)]
pub struct ZarrNode {
    /// Path from store root (e.g., "/" or "/model_a" or "/level_1/level_2")
    pub path: IStr,
    /// Arrays directly in this node (keyed by leaf name, not full path)
    pub arrays: BTreeMap<IStr, ZarrArrayMeta>,
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
        self.path_to_array.get(&path.istr())
    }

    pub fn array_by_path_contains<T: IntoIStr>(
        &self,
        path: T,
    ) -> bool {
        self.path_to_array
            .contains_key(&path.istr())
    }

    pub fn find_paths_matching_prefix<
        T: IntoIStr,
    >(
        &self,
        prefix: T,
    ) -> Option<Vec<IStr>> {
        let binding = prefix.istr();
        let prefix_str: &str = binding.as_ref();
        let matching: Vec<IStr> = self
            .path_to_array
            .keys()
            .cloned()
            .filter(|p| {
                let p_str: &str = p.as_ref();
                p_str.starts_with(prefix_str)
            })
            .collect();

        if matching.is_empty() {
            None
        } else {
            Some(matching)
        }
    }

    /// All data variable paths (flat: just names, hierarchical: includes "group/var" paths)
    pub fn all_data_var_paths(
        &self,
    ) -> Vec<IStr> {
        let mut out = Vec::new();
        self.collect_data_var_paths(
            &self.root, &mut out,
        );
        out
    }

    /// All data variable paths (flat: just names, hierarchical: includes "group/var" paths)
    pub fn all_array_paths(&self) -> Vec<IStr> {
        // self.path_to_array
        //     .keys()
        //     .cloned()
        //     .collect()
        let mut out = Vec::new();
        self.collect_data_var_paths(
            &self.root, &mut out,
        );
        out
        // out.into_iter()
        //     .map(|p| {
        //         p.trim_start_matches('/').istr()
        //     })
        //     .collect()
    }

    pub fn all_zarr_array_paths(
        &self,
    ) -> Vec<IStr> {
        self.path_to_array
            .keys()
            .cloned()
            .collect()
    }

    fn collect_data_var_paths(
        &self,
        node: &ZarrNode,
        out: &mut Vec<IStr>,
    ) {
        // Root node data vars use just the var name
        let path_str: &str = node.path.as_ref();
        let prefix = if path_str == "/" {
            String::new()
        } else {
            format!(
                "{}/",
                path_str.trim_start_matches('/')
            )
        };

        for var in &node.data_vars {
            let path = if prefix.is_empty() {
                var.clone()
            } else {
                format!("{}{}", prefix, var)
                    .istr()
            };
            out.push(path);
        }

        for (_, child) in &node.children {
            self.collect_data_var_paths(
                child, out,
            );
        }
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

        // Add dimension columns (from combined dimension analysis)
        for dim in &self.dim_analysis.all_dims {
            let dtype = self
                .path_to_array
                .get(dim)
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

#[derive(Debug, Clone)]
pub struct ZarrArrayMeta {
    pub path: IStr,
    /// Shape wrapped in Arc for cheap cloning.
    pub shape: Arc<[u64]>,
    /// Regular chunk shape (edge chunks may be smaller).
    pub chunk_shape: Arc<[u64]>,
    pub chunk_grid: Arc<ChunkGrid>,
    pub dims: SmallVec<[IStr; 4]>,
    pub polars_dtype: PlDataType,
    pub time_encoding: Option<TimeEncoding>,
    /// Raw zarrs ArrayMetadata from traverse (for unconsolidated stores)
    pub array_metadata:
        Option<Arc<zarrs::array::ArrayMetadata>>,
}
