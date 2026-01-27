use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use polars::prelude::{DataType as PlDataType, Field, Schema};
use smallvec::SmallVec;

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
    pub path_to_array: BTreeMap<IStr, ZarrArrayMeta>,
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
        let mut node_dims: BTreeMap<IStr, Vec<IStr>> = BTreeMap::new();
        let mut dim_lengths: BTreeMap<IStr, u64> = BTreeMap::new();

        // Collect root dims first (they define primary order)
        let root_dims = root.local_dims.clone();
        for dim in &root_dims {
            if !all_dims.contains(dim) {
                all_dims.push(dim.clone());
            }
        }

        // Recursively collect from all nodes
        Self::collect_node(root, &mut all_dims, &mut node_dims, &mut dim_lengths);

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
        node_dims.insert(node.path.clone(), node.local_dims.clone());

        // Add any new dimensions not yet seen
        for dim in &node.local_dims {
            if !all_dims.contains(dim) {
                all_dims.push(dim.clone());
            }
        }

        // Infer dim lengths from array shapes
        for (_, arr) in &node.arrays {
            for (i, dim) in arr.dims.iter().enumerate() {
                if i < arr.shape.len() {
                    dim_lengths.entry(dim.clone()).or_insert(arr.shape[i]);
                }
            }
        }

        // Recurse into children
        for (_, child) in &node.children {
            Self::collect_node(child, all_dims, node_dims, dim_lengths);
        }
    }

    /// Map a node's local dims to output dim positions.
    /// For each output dim, returns Some(index) if the node has that dim, None otherwise.
    pub fn node_dim_positions(&self, node_path: &str) -> Vec<Option<usize>> {
        let empty = Vec::new();
        let local_dims = self.node_dims.get(&node_path.istr()).unwrap_or(&empty);
        self.all_dims
            .iter()
            .map(|out_dim| local_dims.iter().position(|nd| nd == out_dim))
            .collect()
    }

    /// Check if node shares any dims with root
    pub fn shares_dims_with_root(&self, node_path: &str) -> bool {
        let empty = Vec::new();
        let local_dims = self.node_dims.get(&node_path.istr()).unwrap_or(&empty);
        local_dims.iter().any(|d| self.root_dims.contains(d))
    }

    /// Get the total number of elements for the combined output grid
    pub fn total_elements(&self) -> u64 {
        self.all_dims
            .iter()
            .map(|d| self.dim_lengths.get(d).copied().unwrap_or(1))
            .product()
    }

    /// Get the shape of the combined output grid
    pub fn output_shape(&self) -> Vec<u64> {
        self.all_dims
            .iter()
            .map(|d| self.dim_lengths.get(d).copied().unwrap_or(1))
            .collect()
    }

    /// Compute strides for row-major indexing of the output grid
    pub fn output_strides(&self) -> Vec<u64> {
        let shape = self.output_shape();
        compute_strides(&shape)
    }

    /// Compute the index into a source array given an output row index.
    ///
    /// This handles broadcasting: when the source array has fewer dimensions than
    /// the output, the extra output dimensions are ignored (broadcast/repeated).
    ///
    /// # Arguments
    /// * `output_row` - Row index in the output DataFrame
    /// * `source_dims` - Dimension names for source array (e.g., ["y", "x"])
    /// * `source_shape` - Shape of the source array
    ///
    /// # Returns
    /// Index into the source array's flat buffer
    pub fn compute_source_index(
        &self,
        output_row: u64,
        source_dims: &[IStr],
        source_shape: &[u64],
    ) -> u64 {
        let output_shape = self.output_shape();
        let output_strides = compute_strides(&output_shape);
        let source_strides = compute_strides(source_shape);

        let mut source_idx: u64 = 0;

        for (src_d, src_dim) in source_dims.iter().enumerate() {
            // Find this source dimension in the output dimensions
            if let Some(out_d) = self.all_dims.iter().position(|od| od == src_dim) {
                // Extract coordinate for this dimension from output row
                let coord = (output_row / output_strides[out_d]) % output_shape[out_d];
                if src_d < source_strides.len() {
                    source_idx += coord * source_strides[src_d];
                }
            }
            // If source dim not in output dims, it's an error in metadata
        }

        source_idx
    }
}

/// Compute strides for row-major indexing of an N-dimensional array.
/// 
/// For shape [a, b, c], strides are [b*c, c, 1].
#[inline]
pub fn compute_strides(shape: &[u64]) -> Vec<u64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl ZarrMeta {
    /// True if there are any child groups
    pub fn is_hierarchical(&self) -> bool {
        !self.root.children.is_empty()
    }

    /// Get array meta by path (e.g., "temperature" or "model_a/temperature")
    pub fn array(&self, path: &str) -> Option<&ZarrArrayMeta> {
        self.path_to_array.get(&path.istr())
    }

    /// All data variable paths (flat: just names, hierarchical: includes "group/var" paths)
    pub fn all_data_var_paths(&self) -> Vec<IStr> {
        let mut out = Vec::new();
        self.collect_data_var_paths(&self.root, &mut out);
        out
    }

    fn collect_data_var_paths(&self, node: &ZarrNode, out: &mut Vec<IStr>) {
        // Root node data vars use just the var name
        let path_str: &str = node.path.as_ref();
        let prefix = if path_str == "/" {
            String::new()
        } else {
            format!("{}/", path_str.trim_start_matches('/'))
        };

        for var in &node.data_vars {
            let path = if prefix.is_empty() {
                var.clone()
            } else {
                format!("{}{}", prefix, var).istr()
            };
            out.push(path);
        }

        for (_, child) in &node.children {
            self.collect_data_var_paths(child, out);
        }
    }

    /// Generate a Polars schema for the tidy DataFrame output.
    /// 
    /// For flat datasets, this is the same as ZarrDatasetMeta::tidy_schema.
    /// For hierarchical datasets, child groups become struct columns.
    pub fn tidy_schema(&self, variables: Option<&[IStr]>) -> Schema {
        let var_set: Option<BTreeSet<&str>> =
            variables.map(|v| v.iter().map(|s| s.as_ref()).collect());

        let mut fields: Vec<Field> = Vec::new();

        // Add dimension columns (from combined dimension analysis)
        for dim in &self.dim_analysis.all_dims {
            let dtype = self
                .path_to_array
                .get(dim)
                .map(|m| m.polars_dtype.clone())
                .unwrap_or(PlDataType::Int64);
            let dim_str: &str = dim.as_ref();
            fields.push(Field::new(dim_str.into(), dtype));
        }

        // Add root data variable columns
        for var in &self.root.data_vars {
            let var_str: &str = var.as_ref();
            if var_set.as_ref().map_or(true, |vs| vs.contains(var_str)) {
                if let Some(m) = self.root.arrays.get(var) {
                    fields.push(Field::new(var_str.into(), m.polars_dtype.clone()));
                }
            }
        }

        // Add child group struct columns
        for (child_name, child_node) in &self.root.children {
            let child_name_str: &str = child_name.as_ref();
            // Check if this group or any of its vars are in the selection
            let should_include = var_set.as_ref().map_or(true, |vs| {
                vs.contains(child_name_str)
                    || child_node.data_vars.iter().any(|v| {
                        let v_str: &str = v.as_ref();
                        vs.contains(v_str) || vs.contains(&format!("{}/{}", child_name_str, v_str).as_str())
                    })
            });

            if should_include {
                let struct_dtype = self.node_to_struct_dtype(child_node);
                fields.push(Field::new(child_name_str.into(), struct_dtype));
            }
        }

        fields.into_iter().collect()
    }

    /// Convert a ZarrNode to a Struct dtype for schema generation.
    fn node_to_struct_dtype(&self, node: &ZarrNode) -> PlDataType {
        let mut struct_fields: Vec<Field> = Vec::new();

        // Add data variable fields
        for var in &node.data_vars {
            if let Some(arr_meta) = node.arrays.get(var) {
                let var_str: &str = var.as_ref();
                struct_fields.push(Field::new(var_str.into(), arr_meta.polars_dtype.clone()));
            }
        }

        // Recursively add nested child groups
        for (child_name, child_node) in &node.children {
            let child_name_str: &str = child_name.as_ref();
            let nested_dtype = self.node_to_struct_dtype(child_node);
            struct_fields.push(Field::new(child_name_str.into(), nested_dtype));
        }

        PlDataType::Struct(struct_fields)
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

    /// Recursively iterate over all arrays in this node and descendants
    pub fn all_arrays(&self) -> Box<dyn Iterator<Item = (&IStr, &ZarrArrayMeta)> + '_> {
        let local = self.arrays.iter();
        let children_iter = self.children.values().flat_map(|c| c.all_arrays());
        Box::new(local.chain(children_iter))
    }

    /// Recursively iterate over all nodes (including self)
    pub fn all_nodes(&self) -> Box<dyn Iterator<Item = &ZarrNode> + '_> {
        Box::new(std::iter::once(self).chain(self.children.values().flat_map(|c| c.all_nodes())))
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
            raw * self.unit_ns
        } else {
            self.epoch_ns + raw * self.unit_ns
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZarrArrayMeta {
    pub path: IStr,
    /// Shape wrapped in Arc for cheap cloning.
    pub shape: Arc<[u64]>,
    pub dims: SmallVec<[IStr; 4]>,
    pub polars_dtype: PlDataType,
    pub time_encoding: Option<TimeEncoding>,
}

#[derive(Debug, Clone)]
pub struct ZarrDatasetMeta {
    pub arrays: BTreeMap<IStr, ZarrArrayMeta>,
    pub dims: Vec<IStr>,
    pub data_vars: Vec<IStr>,
}

impl ZarrDatasetMeta {
    pub fn tidy_schema(&self, variables: Option<&[IStr]>) -> Schema {
        let var_set: Option<BTreeSet<&str>> = variables.map(|v| v.iter().map(|s| s.as_ref()).collect());

        let mut fields: Vec<Field> = Vec::new();

        for dim in &self.dims {
            let dtype = self
                .arrays
                .get(dim)
                .map(|m| m.polars_dtype.clone())
                .unwrap_or(PlDataType::Int64);
            fields.push(Field::new((<IStr as AsRef<str>>::as_ref(dim)).into(), dtype));
        }

        let vars_iter: Box<dyn Iterator<Item = &str>> = if let Some(var_set) = &var_set {
            Box::new(
                self.data_vars
                    .iter()
                    .map(|s| s.as_ref())
                    .filter(|v| var_set.contains(v)),
            )
        } else {
            Box::new(self.data_vars.iter().map(|s| s.as_ref()))
        };

        for v in vars_iter {
            if let Some(m) = self.arrays.get(&v.istr()) {
                fields.push(Field::new(v.into(), m.polars_dtype.clone()));
            }
        }

        fields.into_iter().collect()
    }
}


// =============================================================================
// Conversions between ZarrMeta and ZarrDatasetMeta
// =============================================================================

impl From<ZarrDatasetMeta> for ZarrMeta {
    /// Convert a flat ZarrDatasetMeta to the unified ZarrMeta format
    fn from(legacy: ZarrDatasetMeta) -> Self {
        let mut root = ZarrNode::new("/".istr());
        root.arrays = legacy.arrays.clone();
        root.local_dims = legacy.dims.clone();
        root.data_vars = legacy.data_vars.clone();

        let dim_analysis = DimensionAnalysis::compute(&root);

        let mut path_to_array = BTreeMap::new();
        for (name, arr) in &legacy.arrays {
            path_to_array.insert(name.clone(), arr.clone());
        }

        ZarrMeta {
            root,
            dim_analysis,
            path_to_array,
        }
    }
}

impl From<&ZarrMeta> for ZarrDatasetMeta {
    /// Convert a ZarrMeta back to flat ZarrDatasetMeta (loses hierarchy info)
    fn from(meta: &ZarrMeta) -> Self {
        ZarrDatasetMeta {
            arrays: meta.path_to_array.clone(),
            dims: meta.dim_analysis.root_dims.clone(),
            data_vars: meta.root.data_vars.clone(),
        }
    }
}
