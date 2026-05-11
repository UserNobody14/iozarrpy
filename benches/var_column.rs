//! Focused benches for `build_var_column`.
//!
//! Three scenarios that approximate the workloads seen in production:
//! - `fast_path_all`: same-dims, same-chunk, zero offsets, KeepMask::All (zero-copy borrow).
//! - `fast_path_sparse_small`: same-dims, sparse keep (typical filtered query).
//! - `fast_path_sparse_multivar`: realistic multivar shape (1,3,200,200) with sparse 363-row keep.
//! - `slow_path_diff_dims`: different dims with offsets (gather_by hot path).

use std::hint::black_box;
use std::sync::Arc;

use criterion::{
    Criterion, criterion_group, criterion_main,
};

use _core::bench_internals::*;
use polars_arrow::array::PrimitiveArray;

fn f64_column(values: Vec<f64>) -> ColumnData {
    ColumnData::F64(PrimitiveArray::from_vec(values))
}

fn bench_fast_path_all(c: &mut Criterion) {
    let chunk_shape = [10u64, 10, 10];
    let strides = compute_strides(&chunk_shape);
    let chunk_len = 1000usize;
    let keep = KeepMask::All(chunk_len);
    let data: Arc<ColumnData> = Arc::new(f64_column(
        (0..chunk_len).map(|i| i as f64 * 0.1).collect(),
    ));
    let dims: Vec<IStr> = ["x", "y", "time"].into_istrs();
    let offsets = vec![0u64; 3];
    let name = "temperature".istr();

    c.bench_function("build_var_column/fast_path_all_1000", |b| {
        b.iter(|| {
            build_var_column(
                black_box(&name),
                black_box(&data),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&offsets),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&strides),
                black_box(&keep),
                black_box(None),
            )
        })
    });
}

fn bench_fast_path_sparse_small(c: &mut Criterion) {
    let chunk_shape = [10u64, 10, 10];
    let strides = compute_strides(&chunk_shape);
    let chunk_len = 1000usize;
    // Sparse keep: typical 10% of a 10x10x10 chunk
    let idx: Vec<usize> = (0..chunk_len).filter(|i| i % 8 == 0).collect();
    let keep = KeepMask::Sparse(idx);
    let data: Arc<ColumnData> = Arc::new(f64_column(
        (0..chunk_len).map(|i| i as f64 * 0.1).collect(),
    ));
    let dims: Vec<IStr> = ["x", "y", "time"].into_istrs();
    let offsets = vec![0u64; 3];
    let name = "temperature".istr();

    c.bench_function("build_var_column/fast_path_sparse_small_125", |b| {
        b.iter(|| {
            build_var_column(
                black_box(&name),
                black_box(&data),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&offsets),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&strides),
                black_box(&keep),
                black_box(None),
            )
        })
    });
}

fn bench_fast_path_sparse_multivar(c: &mut Criterion) {
    // Multivar grid_default-style chunk: (1, 3, 200, 200) = 120K f64
    let chunk_shape = [1u64, 3, 200, 200];
    let strides = compute_strides(&chunk_shape);
    let chunk_len: usize = chunk_shape.iter().product::<u64>() as usize;
    let dims: Vec<IStr> = ["time", "lead_time", "y", "x"].into_istrs();

    // Keep only y∈[50..61], x∈[100..111] → 1*3*11*11 = 363 rows
    let mut idx: Vec<usize> = Vec::with_capacity(363);
    for t in 0..1 {
        for l in 0..3 {
            for y in 50..61 {
                for x in 100..111 {
                    let row = t * (3 * 200 * 200)
                        + l * (200 * 200)
                        + y * 200
                        + x;
                    idx.push(row);
                }
            }
        }
    }
    assert_eq!(idx.len(), 363);
    let keep = KeepMask::Sparse(idx);
    let data: Arc<ColumnData> = Arc::new(f64_column(
        (0..chunk_len).map(|i| i as f64 * 0.001).collect(),
    ));
    let offsets = vec![0u64; 4];
    let name = "2m_temperature".istr();

    c.bench_function("build_var_column/fast_path_sparse_multivar_363", |b| {
        b.iter(|| {
            build_var_column(
                black_box(&name),
                black_box(&data),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&offsets),
                black_box(&dims),
                black_box(&chunk_shape),
                black_box(&strides),
                black_box(&keep),
                black_box(None),
            )
        })
    });
}

fn bench_slow_path_diff_dims(c: &mut Criterion) {
    let primary_chunk_shape = [10u64, 10, 10];
    let primary_strides = compute_strides(&primary_chunk_shape);
    let primary_dims: Vec<IStr> = ["x", "y", "time"].into_istrs();
    let chunk_len = 1000usize;
    let keep = KeepMask::All(chunk_len);

    let var_dims: Vec<IStr> = ["x", "y"].into_istrs();
    let var_chunk_shape = [20u64, 20];
    let var_data: Arc<ColumnData> = Arc::new(f64_column(
        (0..400).map(|i| i as f64 * 0.01).collect(),
    ));
    let var_offsets = vec![5u64, 3];
    let name = "pressure".istr();

    c.bench_function("build_var_column/slow_path_diff_dims_1000", |b| {
        b.iter(|| {
            build_var_column(
                black_box(&name),
                black_box(&var_data),
                black_box(&var_dims),
                black_box(&var_chunk_shape),
                black_box(&var_offsets),
                black_box(&primary_dims),
                black_box(&primary_chunk_shape),
                black_box(&primary_strides),
                black_box(&keep),
                black_box(None),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_fast_path_all,
    bench_fast_path_sparse_small,
    bench_fast_path_sparse_multivar,
    bench_slow_path_diff_dims,
);
criterion_main!(benches);
