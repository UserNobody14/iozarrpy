use std::collections::BTreeMap;

use std::cmp::Ordering as Ord;

use super::traits::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendSync, HasStore,
};
use crate::errors::BackendError;
use polars::prelude::Expr;

use crate::PlannerStats;
use crate::chunk_plan::GroupedChunkPlan;
use crate::chunk_plan::SyncCoordResolver;
use crate::chunk_plan::compile_expr;
use crate::chunk_plan::selection_to_grouped_chunk_plan_unified_from_meta;
use crate::chunk_plan::{
    MergedCache, collect_requests_with_meta,
    materialize,
};
use crate::meta::ZarrMeta;
use crate::{IStr, chunk_plan::*};

// -----------------------------------------------------------------------------
// Pure helper functions (no I/O, no async - safe to extract without perf impact)
// -----------------------------------------------------------------------------

/// Group resolution requests by dimension for batch processing.
fn group_requests_by_dimension(
    requests: Vec<ResolutionRequest>,
) -> BTreeMap<
    IStr,
    Vec<(ResolutionRequest, ValueRange)>,
> {
    let mut by_dim: BTreeMap<
        IStr,
        Vec<(ResolutionRequest, ValueRange)>,
    > = BTreeMap::new();
    for req in requests {
        by_dim
            .entry(req.dim.clone())
            .or_default()
            .push((
                req.clone(),
                req.value_range.clone(),
            ));
    }
    by_dim
}

/// Binary search: should we move the high bound left? (lower_bound semantics)
#[inline(always)]
fn lower_bound_should_go_left(
    dir: Ord,
    strict: bool,
    cmp: Option<Ord>,
) -> bool {
    matches!(
        (dir, strict, cmp),
        (
            Ord::Less | Ord::Equal,
            false,
            Some(Ord::Greater | Ord::Equal)
        ) | (Ord::Less, true, Some(Ord::Greater))
            | (
                Ord::Greater | Ord::Equal,
                false,
                Some(Ord::Less | Ord::Equal)
            )
            | (
                Ord::Greater,
                true,
                Some(Ord::Less)
            )
    )
}

/// Binary search: should we move the low bound right? (upper_bound semantics)
#[inline(always)]
fn upper_bound_should_go_right(
    dir: Ord,
    strict: bool,
    cmp: Option<Ord>,
) -> bool {
    matches!(
        (dir, strict, cmp),
        (
            Ord::Less | Ord::Equal,
            false,
            Some(Ord::Less | Ord::Equal)
        ) | (Ord::Less, true, Some(Ord::Less))
            | (
                Ord::Greater | Ord::Equal,
                false,
                Some(Ord::Greater | Ord::Equal)
            )
            | (
                Ord::Greater,
                true,
                Some(Ord::Greater)
            )
    )
}

/// Sample indices for monotonicity verification (must be sorted ascending).
fn monotonic_sample_indices(
    n: u64,
    chunk_size: u64,
) -> [u64; 5] {
    let mut samples = [
        0u64,
        chunk_size.saturating_sub(1).min(n - 1),
        chunk_size.min(n - 1),
        (n / 2).min(n - 1),
        n - 1,
    ];
    samples.sort();
    samples
}

/// Check if the observed ordering between consecutive samples matches expected direction.
#[inline(always)]
fn monotonic_ord_matches(
    dir: Ord,
    ord: Option<Ord>,
) -> bool {
    match (dir, ord) {
        (
            Ord::Less,
            Some(Ord::Less | Ord::Equal),
        ) => true,
        (
            Ord::Greater,
            Some(Ord::Greater | Ord::Equal),
        ) => true,
        _ => false,
    }
}

/// Prepare compilation inputs: compile expr to lazy selection and collect resolution requests.
fn prepare_compile_inputs(
    meta: &ZarrMeta,
    expr: &Expr,
) -> Result<
    (
        LazyDatasetSelection,
        Vec<ResolutionRequest>,
        HashMapCache,
    ),
    BackendError,
> {
    let (dims, dim_lengths) =
        compute_dims_and_lengths_unified(meta);
    let vars = meta.all_array_paths();
    let mut ctx =
        LazyCompileCtx::new(&meta, &dims, &vars);
    let lazy_selection =
        compile_expr(expr, &mut ctx)?;
    let (requests, immediate_cache) =
        collect_requests_with_meta(
            &lazy_selection,
            &meta,
            &dim_lengths,
            &dims,
        );
    Ok((
        lazy_selection,
        requests,
        immediate_cache,
    ))
}

/// Materialize lazy selection with resolved cache and convert to grouped chunk plan.
fn finish_compile_with_resolved_cache(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    resolved_cache: &dyn ResolutionCache,
    immediate_cache: &HashMapCache,
) -> Result<
    (GroupedChunkPlan, PlannerStats),
    BackendError,
> {
    let merged = MergedCache::new(
        resolved_cache,
        immediate_cache,
    );
    let selection = materialize(
        lazy_selection,
        &meta,
        &merged,
    )?;
    let stats = PlannerStats { coord_reads: 0 };
    let grouped_plan =
        selection_to_grouped_chunk_plan_unified_from_meta(&selection, meta)?;
    Ok((grouped_plan, stats))
}

/// Compile a Polars expression to a chunk plan synchronously.
pub trait ChunkedExpressionCompilerSync:
    HasMetadataBackendSync<ZarrMeta>
    + ChunkedDataBackendSync
{
    fn compile_expression_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

#[async_trait::async_trait]
pub trait ChunkedExpressionCompilerAsync:
    HasMetadataBackendAsync<ZarrMeta>
    + ChunkedDataBackendAsync
{
    async fn compile_expression_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore
        + SyncCoordResolver,
> ChunkedExpressionCompilerSync for B
{
    fn compile_expression_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata()?;
        let (
            lazy_selection,
            requests,
            immediate_cache,
        ) = prepare_compile_inputs(&meta, expr)?;
        let resolved_cache =
            self.resolve_batch(requests);
        finish_compile_with_resolved_cache(
            &lazy_selection,
            &meta,
            &*resolved_cache,
            &immediate_cache,
        )
    }
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> ChunkedExpressionCompilerAsync for B
{
    async fn compile_expression_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata().await?;
        let (
            lazy_selection,
            requests,
            immediate_cache,
        ) = prepare_compile_inputs(&meta, expr)?;
        let resolved_cache =
            self.resolve_batch(requests).await;
        finish_compile_with_resolved_cache(
            &lazy_selection,
            &meta,
            &*resolved_cache,
            &immediate_cache,
        )
    }
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> AsyncCoordResolver for B
{
    async fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        let mut cache = HashMapCache::new();
        let by_dim =
            group_requests_by_dimension(requests);

        for (dim, reqs) in by_dim {
            let results = self
                .resolve_dimension(&dim, reqs)
                .await;
            for (req, result) in results {
                cache.insert(req, result);
            }
        }

        Box::new(cache)
    }
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore
        + Sync,
> SyncCoordResolver for B
{
    fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        use rayon::prelude::*;

        let mut cache = HashMapCache::new();
        let by_dim =
            group_requests_by_dimension(requests);

        let results: Vec<_> = by_dim
            .into_par_iter()
            .flat_map_iter(|(dim, reqs)| {
                self.resolve_dimension_sync(
                    &dim, reqs,
                )
            })
            .collect();

        for (req, result) in results {
            cache.insert(req, result);
        }

        Box::new(cache)
    }
}

trait ResolveDimension {
    async fn resolve_dimension(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRange,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )>;

    async fn scalar_at(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar>;

    async fn check_monotonic(
        &self,
        dim: &IStr,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<Ord>;

    async fn lower_bound(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64>;

    async fn upper_bound(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64>;

    async fn resolve_range(
        &self,
        dim: &IStr,
        vr: &ValueRange,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<std::ops::Range<u64>>;
}

trait ResolveDimensionSync {
    fn resolve_dimension_sync(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRange,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )>;

    fn scalar_at_sync(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar>;

    fn check_monotonic_sync(
        &self,
        dim: &IStr,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<Ord>;

    fn lower_bound_sync(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64>;

    fn upper_bound_sync(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64>;

    fn resolve_range_sync(
        &self,
        dim: &IStr,
        vr: &ValueRange,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<std::ops::Range<u64>>;
}

impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> ResolveDimension for B
{
    async fn resolve_dimension(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRange,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )> {
        // Get coordinate array metadata
        let coord_meta = match self
            .metadata()
            .await
            .ok()
            .and_then(|meta| {
                meta.array_by_path(dim.clone())
                    .cloned()
            }) {
            Some(m) => m,
            None => {
                return reqs
                    .into_iter()
                    .map(|(req, _)| (req, None))
                    .collect();
            }
        };

        if coord_meta.shape.len() != 1 {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, None))
                .collect();
        }

        let n = coord_meta.shape[0];
        if n == 0 {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, Some(0..0)))
                .collect();
        }

        let chunk_size = coord_meta
            .chunk_shape
            .first()
            .copied()
            .unwrap_or(n);
        let time_enc =
            coord_meta.time_encoding.clone();

        // Use the full array path from metadata for I/O operations.
        // The dim name alone (e.g. "y") is insufficient when the zarr store
        // root is not "/" — we need the full path (e.g. "/forecasts/ds.zarr/y")
        // so that Array::async_open can locate the array in the store.
        let array_path = coord_meta.path.clone();

        // Check monotonicity
        let Some(dir) = self
            .check_monotonic(
                &array_path,
                n,
                chunk_size,
                time_enc.as_ref(),
            )
            .await
        else {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, None))
                .collect();
        };

        let time_enc2 = time_enc.clone();

        use futures::future::join_all;

        let futures =
            reqs.into_iter().map(|(req, vr)| {
                let this = self;
                let time_enc3 = time_enc2.clone();
                let array_path2 =
                    array_path.clone();
                async move {
                    let result = this
                        .resolve_range(
                            &array_path2,
                            &vr,
                            dir,
                            n,
                            chunk_size,
                            time_enc3.as_ref(),
                        )
                        .await;
                    (req, result)
                }
            });

        let results = join_all(futures).await;

        results
    }

    async fn scalar_at(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar> {
        if idx >= n {
            return None;
        }
        let chunk_idx = idx / chunk_size;
        let offset = (idx % chunk_size) as usize;

        let chunk = self
            .read_chunk_async(dim, &[chunk_idx])
            .await
            .ok()?;

        chunk
            .get_i64(offset)
            .map(|raw| {
                crate::chunk_plan::apply_time_encoding(
                    raw, time_enc,
                )
            })
    }

    async fn check_monotonic(
        &self,
        dim: &IStr,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<Ord> {
        if n < 2 {
            return Some(Ord::Less);
        }

        let first = self
            .scalar_at(
                dim, 0, n, chunk_size, time_enc,
            )
            .await?;
        let last = self
            .scalar_at(
                dim,
                n - 1,
                n,
                chunk_size,
                time_enc,
            )
            .await?;

        let dir = match first.partial_cmp(&last) {
            Some(Ord::Less | Ord::Equal) => {
                Ord::Less
            }
            Some(Ord::Greater) => Ord::Greater,
            None => return None,
        };

        let samples = monotonic_sample_indices(
            n, chunk_size,
        );
        let mut prev: Option<CoordScalar> = None;
        for &i in &samples {
            let v = self
                .scalar_at(
                    dim, i, n, chunk_size,
                    time_enc,
                )
                .await?;
            if let Some(p) = &prev {
                if !monotonic_ord_matches(
                    dir,
                    p.partial_cmp(&v),
                ) {
                    return None;
                }
            }
            prev = Some(v);
        }

        Some(dir)
    }

    async fn lower_bound(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self
                .scalar_at(
                    dim, mid, n, chunk_size,
                    time_enc,
                )
                .await?;
            let cmp = v.partial_cmp(target);
            let go_left =
                lower_bound_should_go_left(
                    dir, strict, cmp,
                );

            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        Some(lo)
    }

    async fn upper_bound(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self
                .scalar_at(
                    dim, mid, n, chunk_size,
                    time_enc,
                )
                .await?;
            let cmp = v.partial_cmp(target);
            let go_right =
                upper_bound_should_go_right(
                    dir, strict, cmp,
                );

            if go_right {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Some(lo)
    }

    async fn resolve_range(
        &self,
        dim: &IStr,
        vr: &ValueRange,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<std::ops::Range<u64>> {
        use std::ops::Bound;
        let vr = vr.as_ref()?;
        let start = match &vr.0 {
            Bound::Included(s) => {
                self.lower_bound(
                    dim, s, false, dir, n,
                    chunk_size, time_enc,
                )
                .await?
            }
            Bound::Excluded(s) => {
                self.lower_bound(
                    dim, s, true, dir, n,
                    chunk_size, time_enc,
                )
                .await?
            }
            Bound::Unbounded => 0,
        };
        let end = match &vr.1 {
            Bound::Included(s) => {
                self.upper_bound(
                    dim, s, false, dir, n,
                    chunk_size, time_enc,
                )
                .await?
            }
            Bound::Excluded(s) => {
                self.upper_bound(
                    dim, s, true, dir, n,
                    chunk_size, time_enc,
                )
                .await?
            }
            Bound::Unbounded => n,
        };
        Some(start..end)
    }
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore
        + Sync,
> ResolveDimensionSync for B
{
    fn resolve_dimension_sync(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRange,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )> {
        use rayon::prelude::*;

        // Get coordinate array metadata
        let coord_meta = match self
            .metadata()
            .ok()
            .and_then(|meta| {
                meta.clone()
                    .array_by_path(dim.clone())
                    .cloned()
            }) {
            Some(m) => m,
            None => {
                return reqs
                    .into_iter()
                    .map(|(req, _)| (req, None))
                    .collect();
            }
        };

        if coord_meta.shape.len() != 1 {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, None))
                .collect();
        }

        let n = coord_meta.shape[0];
        if n == 0 {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, Some(0..0)))
                .collect();
        }

        let chunk_size = coord_meta
            .chunk_shape
            .first()
            .copied()
            .unwrap_or(n);
        let time_enc =
            coord_meta.time_encoding.clone();

        // Use the full array path from metadata for I/O operations.
        // The dim name alone (e.g. "y") is insufficient when the zarr store
        // root is not "/" — we need the full path (e.g. "/forecasts/ds.zarr/y")
        // so that Array::async_open can locate the array in the store.
        let array_path = coord_meta.path.clone();

        // Check monotonicity
        let Some(dir) = self
            .check_monotonic_sync(
                &array_path,
                n,
                chunk_size,
                time_enc.as_ref(),
            )
        else {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, None))
                .collect();
        };

        reqs.into_par_iter()
            .map(|(req, vr)| {
                let resolved = self
                    .resolve_range_sync(
                        &array_path,
                        &vr,
                        dir,
                        n,
                        chunk_size,
                        time_enc.as_ref(),
                    );
                (req, resolved)
            })
            .collect()
    }

    fn scalar_at_sync(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar> {
        if idx >= n {
            return None;
        }
        let chunk_idx = idx / chunk_size;
        let offset = (idx % chunk_size) as usize;

        let chunk = self
            .read_chunk_sync(dim, &[chunk_idx])
            .ok()?;

        chunk
            .get_i64(offset)
            .map(|raw| {
                crate::chunk_plan::apply_time_encoding(
                    raw, time_enc,
                )
            })
    }

    fn check_monotonic_sync(
        &self,
        dim: &IStr,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<Ord> {
        if n < 2 {
            return Some(Ord::Less);
        }

        let first = self.scalar_at_sync(
            dim, 0, n, chunk_size, time_enc,
        )?;
        let last = self.scalar_at_sync(
            dim,
            n - 1,
            n,
            chunk_size,
            time_enc,
        )?;

        let dir = match first.partial_cmp(&last) {
            Some(Ord::Less | Ord::Equal) => {
                Ord::Less
            }
            Some(Ord::Greater) => Ord::Greater,
            None => return None,
        };

        let samples = monotonic_sample_indices(
            n, chunk_size,
        );
        let mut prev: Option<CoordScalar> = None;
        for &i in &samples {
            let v = self.scalar_at_sync(
                dim, i, n, chunk_size, time_enc,
            )?;
            if let Some(p) = &prev {
                if !monotonic_ord_matches(
                    dir,
                    p.partial_cmp(&v),
                ) {
                    return None;
                }
            }
            prev = Some(v);
        }

        Some(dir)
    }

    fn lower_bound_sync(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at_sync(
                dim, mid, n, chunk_size, time_enc,
            )?;
            let cmp = v.partial_cmp(target);
            let go_left =
                lower_bound_should_go_left(
                    dir, strict, cmp,
                );

            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        Some(lo)
    }

    fn upper_bound_sync(
        &self,
        dim: &IStr,
        target: &CoordScalar,
        strict: bool,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at_sync(
                dim, mid, n, chunk_size, time_enc,
            )?;
            let cmp = v.partial_cmp(target);
            let go_right =
                upper_bound_should_go_right(
                    dir, strict, cmp,
                );

            if go_right {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Some(lo)
    }

    fn resolve_range_sync(
        &self,
        dim: &IStr,
        vr: &ValueRange,
        dir: Ord,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<std::ops::Range<u64>> {
        if let Some(vr) = vr {
            use crate::chunk_plan::HasCoordBound;
            let lower = |b: &CoordBound| {
                b.get_scalar().and_then(|t| {
                    self.lower_bound_sync(
                        dim,
                        &t,
                        b.is_exclusive(),
                        dir,
                        n,
                        chunk_size,
                        time_enc,
                    )
                })
            };
            let upper = |b: &CoordBound| {
                b.get_scalar().and_then(|t| {
                    self.upper_bound_sync(
                        dim,
                        &t,
                        b.is_exclusive(),
                        dir,
                        n,
                        chunk_size,
                        time_enc,
                    )
                })
            };
            compute_bounds_from_value_range(
                vr, n, lower, upper,
            )
            .map(|(s, e)| s..e)
        } else {
            Some(0..0)
        }
    }
}
