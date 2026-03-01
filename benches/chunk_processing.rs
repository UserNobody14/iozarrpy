use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use std::hint::black_box;

use criterion::{
    Criterion, criterion_group, criterion_main,
};
use polars::prelude::*;
use smallvec::SmallVec;
use zarrs::array::chunk_grid::regular::RegularChunkGrid;
use zarrs::array::ChunkGrid;

use _core::bench_internals::*;
use _core::{IStr, IntoIStr};

// =============================================================================
// Mock backend
// =============================================================================

struct MockBackendSync {
    coord_chunk: Arc<ColumnData>,
    var_chunk: Arc<ColumnData>,
    coord_names: Vec<IStr>,
}

impl MockBackendSync {
    fn new(coord_chunk_len: usize, var_chunk_len: usize, coord_names: &[&str]) -> Self {
        Self {
            coord_chunk: Arc::new(ColumnData::F64(
                (0..coord_chunk_len).map(|i| i as f64).collect(),
            )),
            var_chunk: Arc::new(ColumnData::F64(
                (0..var_chunk_len).map(|i| i as f64 * 0.1).collect(),
            )),
            coord_names: coord_names.iter().map(|s| s.istr()).collect(),
        }
    }
}

impl ChunkedDataBackendSync for MockBackendSync {
    fn read_chunk_sync(
        &self,
        var: &IStr,
        _chunk_idx: &[u64],
    ) -> Result<Arc<ColumnData>, BackendError> {
        if self.coord_names.contains(var) {
            Ok(self.coord_chunk.clone())
        } else {
            Ok(self.var_chunk.clone())
        }
    }
}

// =============================================================================
// Test metadata builder
// =============================================================================

fn make_chunk_grid(
    array_shape: &[u64],
    chunk_shape: &[u64],
) -> Arc<ChunkGrid> {
    let cs: Vec<NonZeroU64> = chunk_shape
        .iter()
        .map(|&s| NonZeroU64::new(s).unwrap())
        .collect();
    Arc::new(ChunkGrid::new(
        RegularChunkGrid::new(array_shape.to_vec(), cs).unwrap(),
    ))
}

fn make_array_meta(
    path: &str,
    dims: &[&str],
    shape: &[u64],
    chunk_shape: &[u64],
    dtype: DataType,
) -> (IStr, Arc<ZarrArrayMeta>) {
    let dim_sv: SmallVec<[IStr; 4]> =
        dims.iter().map(|d| d.istr()).collect();
    let cg = make_chunk_grid(shape, chunk_shape);
    let meta = ZarrArrayMeta {
        path: path.istr(),
        shape: shape.into(),
        chunk_shape: chunk_shape.into(),
        chunk_grid: cg,
        dims: dim_sv,
        polars_dtype: dtype,
        encoding: None,
        array_metadata: None,
    };
    (path.istr(), Arc::new(meta))
}

fn make_test_meta() -> ZarrMeta {
    let mut arrays: BTreeMap<IStr, Arc<ZarrArrayMeta>> = BTreeMap::new();
    let mut path_to_array: BTreeMap<IStr, Arc<ZarrArrayMeta>> = BTreeMap::new();

    // 1D coordinate arrays
    let coords = [
        ("x", 100u64, 10u64),
        ("y", 100, 10),
        ("time", 50, 10),
    ];
    for (name, len, cs) in &coords {
        let (key, meta) =
            make_array_meta(name, &[name], &[*len], &[*cs], DataType::Float64);
        arrays.insert(key.clone(), meta.clone());
        path_to_array.insert(key, meta);
    }

    // 3D data variable — same chunk shape as primary grid
    let (key, meta) = make_array_meta(
        "temperature",
        &["x", "y", "time"],
        &[100, 100, 50],
        &[10, 10, 10],
        DataType::Float64,
    );
    arrays.insert(key.clone(), meta.clone());
    path_to_array.insert(key, meta);

    // 3D data variable — different chunk shape
    let (key, meta) = make_array_meta(
        "pressure",
        &["x", "y", "time"],
        &[100, 100, 50],
        &[20, 20, 5],
        DataType::Float64,
    );
    arrays.insert(key.clone(), meta.clone());
    path_to_array.insert(key, meta);

    let root = ZarrNode {
        path: "/".istr(),
        arrays,
        children: BTreeMap::new(),
        local_dims: vec!["x".istr(), "y".istr(), "time".istr()],
        data_vars: vec!["temperature".istr(), "pressure".istr()],
    };

    let dim_analysis = DimensionAnalysis::compute(&root);

    ZarrMeta {
        root,
        dim_analysis,
        path_to_array,
    }
}

// =============================================================================
// Benchmark: compute_in_bounds_mask
// =============================================================================

fn bench_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_in_bounds_mask");

    let chunk_shape = [10u64, 10, 10];
    let array_shape = [100u64, 100, 50];
    let strides = compute_strides(&chunk_shape);
    let chunk_len = 1000usize; // 10*10*10

    // Interior chunk — O(ndim) fast path
    let interior_origin = [20u64, 30, 10];
    group.bench_function("interior", |b| {
        b.iter(|| {
            compute_in_bounds_mask(
                black_box(chunk_len),
                black_box(&chunk_shape),
                black_box(&interior_origin),
                black_box(&array_shape),
                black_box(&strides),
                None,
            )
        })
    });

    // Edge chunk — O(chunk_len * ndim) slow path
    let edge_origin = [90u64, 90, 40];
    group.bench_function("edge", |b| {
        b.iter(|| {
            compute_in_bounds_mask(
                black_box(chunk_len),
                black_box(&chunk_shape),
                black_box(&edge_origin),
                black_box(&array_shape),
                black_box(&strides),
                None,
            )
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: build_coord_column
// =============================================================================

fn bench_coord_column(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_coord_column");

    let chunk_shape = [10u64, 10, 10];
    let strides = compute_strides(&chunk_shape);
    let origin = [20u64, 30, 10];
    let chunk_len = 1000usize;

    let keep_all = KeepMask::All(chunk_len);

    // Coord data available
    let coord_data =
        ColumnData::F64((0..10).map(|i| 20.0 + i as f64).collect());

    group.bench_function("all_with_coord_data", |b| {
        b.iter(|| {
            build_coord_column(
                black_box("x"),
                black_box(0),
                black_box(&keep_all),
                black_box(&strides),
                black_box(&chunk_shape),
                black_box(&origin),
                black_box(Some(&coord_data)),
                black_box(None),
            )
        })
    });

    // Sparse keep mask with coord data
    let sparse_indices: Vec<usize> = (0..chunk_len).filter(|i| i % 3 == 0).collect();
    let keep_sparse = KeepMask::Sparse(sparse_indices);

    group.bench_function("sparse_with_coord_data", |b| {
        b.iter(|| {
            build_coord_column(
                black_box("x"),
                black_box(0),
                black_box(&keep_sparse),
                black_box(&strides),
                black_box(&chunk_shape),
                black_box(&origin),
                black_box(Some(&coord_data)),
                black_box(None),
            )
        })
    });

    // No coord data (integer fallback)
    group.bench_function("all_no_coord_data", |b| {
        b.iter(|| {
            build_coord_column(
                black_box("x"),
                black_box(0),
                black_box(&keep_all),
                black_box(&strides),
                black_box(&chunk_shape),
                black_box(&origin),
                black_box(None),
                black_box(None),
            )
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: build_var_column
// =============================================================================

fn bench_var_column(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_var_column");

    let chunk_shape = [10u64, 10, 10];
    let strides = compute_strides(&chunk_shape);
    let chunk_len = 1000usize;
    let keep_all = KeepMask::All(chunk_len);

    let var_data: Arc<ColumnData> =
        Arc::new(ColumnData::F64((0..chunk_len).map(|i| i as f64 * 0.1).collect()));

    let dims: Vec<IStr> = vec!["x".istr(), "y".istr(), "time".istr()];
    let var_dims_same = dims.clone();
    let offsets_zero = vec![0u64; 3];

    // Same dims fast path (zero-copy when All)
    group.bench_function("same_dims_fast_path", |b| {
        b.iter(|| {
            build_var_column(
                black_box(&"temperature".istr()),
                black_box(var_data.clone()),
                black_box(&var_dims_same),
                black_box(&chunk_shape),
                black_box(&offsets_zero),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&strides),
                black_box(&keep_all),
                black_box(None),
            )
        })
    });

    // Different dims slow path (gather_by)
    let var_dims_diff: Vec<IStr> = vec!["x".istr(), "y".istr()];
    let var_chunk_shape_diff = [20u64, 20];
    let var_data_diff: Arc<ColumnData> =
        Arc::new(ColumnData::F64((0..400).map(|i| i as f64 * 0.01).collect()));
    let var_offsets_diff = vec![5u64, 3];

    group.bench_function("diff_dims_slow_path", |b| {
        b.iter(|| {
            build_var_column(
                black_box(&"pressure".istr()),
                black_box(var_data_diff.clone()),
                black_box(&var_dims_diff),
                black_box(&var_chunk_shape_diff),
                black_box(&var_offsets_diff),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&strides),
                black_box(&keep_all),
                black_box(None),
            )
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: compute_var_chunk_indices
// =============================================================================

fn bench_var_indices(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_var_chunk_indices");

    let primary_idx = [5u64, 3, 2];
    let primary_chunk_shape = [10u64, 10, 10];
    let primary_dims: Vec<IStr> = vec!["x".istr(), "y".istr(), "time".istr()];

    // Same dimensions
    let var_dims_same = primary_dims.clone();
    let var_chunk_shape_same = [10u64, 10, 10];
    let var_shape_same = [100u64, 100, 50];

    group.bench_function("same_dims", |b| {
        b.iter(|| {
            compute_var_chunk_indices(
                black_box(&primary_idx),
                black_box(&primary_chunk_shape),
                black_box(&primary_dims),
                black_box(&var_dims_same),
                black_box(&var_chunk_shape_same),
                black_box(&var_shape_same),
            )
        })
    });

    // Different dimensions
    let var_dims_diff: Vec<IStr> = vec!["x".istr(), "time".istr()];
    let var_chunk_shape_diff = [20u64, 5];
    let var_shape_diff = [100u64, 50];

    group.bench_function("diff_dims", |b| {
        b.iter(|| {
            compute_var_chunk_indices(
                black_box(&primary_idx),
                black_box(&primary_chunk_shape),
                black_box(&primary_dims),
                black_box(&var_dims_diff),
                black_box(&var_chunk_shape_diff),
                black_box(&var_shape_diff),
            )
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: chunk_to_df (sync, full pipeline)
// =============================================================================

fn bench_chunk_to_df(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_to_df_sync");

    let meta = make_test_meta();
    let backend = MockBackendSync::new(10, 1000, &["x", "y", "time"]);

    let dims_sv: SmallVec<[IStr; 4]> =
        vec!["x".istr(), "y".istr(), "time".istr()].into();
    let cs_sv: SmallVec<[u64; 4]> = vec![10u64, 10, 10].into();
    let sig = ChunkGridSignature::new(dims_sv, cs_sv);
    let array_shape = [100u64, 100, 50];
    let vars: Vec<IStr> = vec!["temperature".istr()];

    // Interior chunk
    group.bench_function("interior_chunk", |b| {
        b.iter(|| {
            chunk_to_df_from_grid_with_backend(
                black_box(&backend),
                black_box(vec![2, 3, 1]),
                black_box(&sig),
                black_box(&array_shape),
                black_box(&vars),
                black_box(None),
                black_box(None),
                black_box(&meta),
            )
            .unwrap()
        })
    });

    // Edge chunk
    group.bench_function("edge_chunk", |b| {
        b.iter(|| {
            chunk_to_df_from_grid_with_backend(
                black_box(&backend),
                black_box(vec![9, 9, 4]),
                black_box(&sig),
                black_box(&array_shape),
                black_box(&vars),
                black_box(None),
                black_box(None),
                black_box(&meta),
            )
            .unwrap()
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: compile_expr
// =============================================================================

fn bench_compile_expr(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_expr");

    let meta = make_test_meta();
    let (dims, _) = compute_dims_and_lengths_unified(&meta);

    // Simple column reference
    let simple_expr = col("temperature");
    group.bench_function("column_ref", |b| {
        b.iter(|| {
            let mut ctx = LazyCompileCtx::new(&meta, &dims);
            compile_expr(black_box(&simple_expr), black_box(&mut ctx)).unwrap()
        })
    });

    // Range comparison: x > 10
    let range_expr = col("x").gt(lit(10i64));
    group.bench_function("range_cmp", |b| {
        b.iter(|| {
            let mut ctx = LazyCompileCtx::new(&meta, &dims);
            compile_expr(black_box(&range_expr), black_box(&mut ctx)).unwrap()
        })
    });

    // Compound: x > 10 & y < 50
    let compound_expr = col("x").gt(lit(10i64)).and(col("y").lt(lit(50i64)));
    group.bench_function("compound_and", |b| {
        b.iter(|| {
            let mut ctx = LazyCompileCtx::new(&meta, &dims);
            compile_expr(black_box(&compound_expr), black_box(&mut ctx)).unwrap()
        })
    });

    // Complex: (x > 10 & y < 50) | (time >= 20)
    let complex_expr = col("x")
        .gt(lit(10i64))
        .and(col("y").lt(lit(50i64)))
        .or(col("time").gt_eq(lit(20i64)));
    group.bench_function("complex_or", |b| {
        b.iter(|| {
            let mut ctx = LazyCompileCtx::new(&meta, &dims);
            compile_expr(black_box(&complex_expr), black_box(&mut ctx)).unwrap()
        })
    });

    group.finish();
}

// =============================================================================
// Benchmark: selection_to_grouped_chunk_plan
// =============================================================================

fn bench_selection_to_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection_to_plan");

    let meta = make_test_meta();

    // NoSelectionMade — creates plans covering all chunks
    let no_selection = DatasetSelection::NoSelectionMade;
    group.bench_function("no_selection_made", |b| {
        b.iter(|| {
            selection_to_grouped_chunk_plan_unified_from_meta(
                black_box(&no_selection),
                black_box(&meta),
            )
            .unwrap()
        })
    });

    group.finish();
}

// =============================================================================
// Criterion harness
// =============================================================================

criterion_group!(
    benches,
    bench_mask,
    bench_coord_column,
    bench_var_column,
    bench_var_indices,
    bench_chunk_to_df,
    bench_compile_expr,
    bench_selection_to_plan,
);
criterion_main!(benches);
