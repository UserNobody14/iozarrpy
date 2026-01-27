---
name: Remove Primary Variable Hack
overview: Redesign the chunk planning system with a per-chunk-grid architecture analogous to GroupedSelection, deriving dimensions from metadata and properly handling heterogeneous chunk layouts.
todos:
  - id: chunk-grid-signature
    content: Switch to ChunkGridSignature type in types.rs (dims + chunk_shape)
    status: completed
  - id: grouped-chunk-plan
    content: Add GroupedChunkPlan type that groups variables by chunk grid signature
    status: completed
  - id: compute-dims
    content: Add compute_dims_and_lengths helper to compile_entry.rs
    status: completed
  - id: update-compile-entry
    content: Remove primary_var from compile APIs, return DatasetSelection
    status: completed
  - id: fix-metadata-loading
    content: Use open_and_load_zarr_meta for hierarchical path support
    status: completed
  - id: update-open-arrays
    content: Restructure open_arrays.rs to remove 'primary' concept
    status: completed
  - id: update-selected-chunks
    content: Update selected_chunks.rs to use dataset selection APIs
    status: completed
  - id: update-debug
    content: Update debug.rs to use hierarchical metadata loading
    status: completed
  - id: update-backend-py
    content: Update backend/py.rs caching to use hierarchical metadata
    status: completed
  - id: test-fixes
    content: All 20 predicate pushdown tests now passing
    status: completed
  - id: expr-advanced-tests
    content: All expr_advanced tests now passing (20 passed, 9 xfailed)
    status: completed
  - id: selection-to-grouped-plan
    content: Added selection_to_grouped_chunk_plan function
    status: completed
  - id: grouped-plan-entry-points
    content: Added compile_expr_to_grouped_chunk_plan entry points
    status: completed
  - id: heterogeneous-chunk-tests
    content: Added tests for heterogeneous chunk grids
    status: completed
  - id: datatree-struct-repr
    content: "Future: Full datatree struct representation (7 E2E tests still pending)"
    status: pending
  - id: scan-loop-per-grid
    content: "Future: Update scan loop to iterate per chunk grid (needed for edge cases)"
    status: pending
isProject: false
---

# Remove Primary Variable Hack - Per-Chunk-Grid Architecture

## Problem Summary

The current design requires a "primary variable" that:

1. Arbitrarily picks `vars[0]` to extract dims/dim_lengths and chunk grid
2. Fails for hierarchical zarr paths (e.g., `model_a/temperature`)
3. Assumes all variables share the same chunk grid (incorrect for heterogeneous datasets)
4. Doesn't reflect the actual internal structure

## New Architecture

### Group by Chunk Grid and Dim Signature

Following the pattern established by `GroupedSelection` (which groups by `DimSignature`), we introduce a second level of grouping for chunk iteration:

```
Expr → LazyDatasetSelection → DatasetSelection
            ↓                      ↓
       by DimSignature and Chunk Grid
```

**Key insight**: Variables with same dimensions but different chunk layouts should have different `DataArraySelection` s.

### Modified Types

#### 1. DimSignature -> ChunkGridSignature (`[src/chunk_plan/indexing/types.rs](src/chunk_plan/indexing/types.rs)`)

```rust
/// Chunk grid signature - dimensions + chunk shape for grouping.
/// 
/// Variables with the same chunk grid can share chunk iteration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkGridSignature {
    /// Dimension names (same semantics as DimSignature)
    dims: SmallVec<[IStr; 4]>,
    /// Chunk shape per dimension (determines grid layout)
    chunk_shape: SmallVec<[u64; 4]>,
}
```

#### 2. GroupedChunkPlan (`[src/chunk_plan/indexing/plan.rs](src/chunk_plan/indexing/plan.rs)`)

```rust
/// Grouped chunk plan - maps chunk grid signatures to plans.
/// 
/// Analogous to GroupedSelection but for chunk iteration.
pub struct GroupedChunkPlan {
    /// ChunkPlan by grid signature
    by_grid: BTreeMap<Arc<ChunkGridSignature>, ChunkPlan>,
    /// Variable name to grid signature lookup
    var_to_grid: BTreeMap<IStr, Arc<ChunkGridSignature>>,
}

impl GroupedChunkPlan {
    /// Iterate over (grid_signature, variables, chunk_plan) tuples
    pub fn iter_grids(&self) -> impl Iterator<Item = (&ChunkGridSignature, Vec<&IStr>, &ChunkPlan)>
    
    /// Get all variables for a grid signature
    pub fn vars_for_grid(&self, sig: &ChunkGridSignature) -> Vec<&IStr>
    
    /// Total number of chunks across all grids
    pub fn total_chunks(&self) -> usize
}
```

### Compute dims/dim_lengths from Metadata

```rust
// In compile_entry.rs
fn compute_dims_and_lengths(meta: &ZarrDatasetMeta) -> (Vec<IStr>, Vec<u64>) {
    let dims = meta.dims.clone();
    let dim_lengths: Vec<u64> = dims.iter().map(|d| {
        meta.arrays.get(d)
            .and_then(|a| a.shape.first().copied())
            .unwrap_or(1)
    }).collect();
    (dims, dim_lengths)
}
```

### API Changes

#### compile_entry.rs - Remove primary_var

| Current | New |

|---------|-----|

| `compile_expr_to_lazy_selection(expr, meta, primary_var)` | `compile_expr_to_lazy_selection(expr, meta)` |

| `resolve_lazy_selection_sync(lazy, meta, store, primary_var)` | `resolve_lazy_selection_sync(lazy, meta, store)` |

| `compile_expr_to_dataset_selection(expr, meta, store, primary_var)` | `compile_expr_to_dataset_selection(expr, meta, store)` |

| `compile_expr_to_chunk_plan(expr, meta, store, primary_var)` | **Remove** - use GroupedChunkPlan instead |

#### New Entry Point

```rust
/// Compile expression to grouped chunk plan (handles heterogeneous grids).
pub fn compile_expr_to_grouped_chunk_plan(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: ReadableWritableListableStorage,
) -> Result<(GroupedChunkPlan, PlannerStats), CompileError>
```

### Scan Loop Redesign

Current (single grid):

```rust
for idx in plan.into_index_iter() {
    chunk_to_df(idx, primary, ...)
}
```

New (per-grid iteration):

```rust
for (grid_sig, vars, plan) in grouped_plan.iter_grids() {
    for idx in plan.into_index_iter() {
        chunk_to_df_for_grid(idx, grid_sig, vars, var_arrays, coord_arrays, ...)
    }
}
```

### open_arrays Redesign

Remove the "primary" return value:

```rust
// Before
pub async fn open_arrays_async(...) -> Result<(
    Arc<Array<...>>,                    // "primary" - REMOVE
    Vec<(IStr, Arc<Array<...>>)>,       // var_arrays
    Vec<(IStr, Arc<Array<...>>)>,       // coord_arrays
), String>

// After  
pub async fn open_arrays_async(...) -> Result<(
    Vec<(IStr, Arc<Array<...>>)>,       // var_arrays
    Vec<(IStr, Arc<Array<...>>)>,       // coord_arrays
), String>
```

### chunk_to_df Adaptation

The `chunk_to_df` function currently takes a "primary" array for geometry. It will instead:

1. Take a `ChunkGridSignature` to identify which grid this chunk belongs to
2. Use any array from that grid group for geometry (they all share the same grid)

## Files to Modify

| File | Changes |

|------|---------|

| `[src/chunk_plan/indexing/types.rs](src/chunk_plan/indexing/types.rs)` | Switch to `ChunkGridSignature` |

| `[src/chunk_plan/indexing/plan.rs](src/chunk_plan/indexing/plan.rs)` | Add `GroupedChunkPlan` |

| `[src/chunk_plan/compile_entry.rs](src/chunk_plan/compile_entry.rs)` | Remove `primary_var`, add `compute_dims_and_lengths`, add `compile_expr_to_grouped_chunk_plan` |

| `[src/chunk_plan/mod.rs](src/chunk_plan/mod.rs)` | Export new types |

| `[src/scan/open_arrays.rs](src/scan/open_arrays.rs)` | Remove "primary" return |

| `[src/scan/chunk_to_df.rs](src/scan/chunk_to_df.rs)` | Adapt to grid-based iteration |

| `[src/scan/scan_async.rs](src/scan/scan_async.rs)` | Use GroupedChunkPlan |

| `[src/py/selected_chunks.rs](src/py/selected_chunks.rs)` | Use new APIs |

| `[src/py/debug.rs](src/py/debug.rs)` | Update internal calls |

| `[src/backend/py.rs](src/backend/py.rs)` | Use new scan architecture |

## Benefits

1. **Correct semantics**: No arbitrary "primary" variable selection
2. **Heterogeneous chunking**: Variables with different chunk layouts work correctly
3. **Hierarchical support**: Path lookups work for datatree structures
4. **Consistent design**: Follows the GroupedSelection pattern
5. **Efficient**: Variables sharing a grid share chunk iteration

## Testing

The 24 failing `test_datatree_predicate_pushdown.py` tests should pass. Additionally, we should add tests for:

- 2d Interpolation where the 2 variables have different chunk shapes
- Two variables with same dims but different chunk shapes
- Mixed hierarchical and flat variables

