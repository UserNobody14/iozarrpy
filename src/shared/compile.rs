use std::collections::BTreeMap;
use std::sync::Arc;

use polars::prelude::Expr;

use super::traits::{
    BackendError, ChunkedDataBackendAsync,
    ChunkedDataBackendSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendSync, HasStore,
};

use crate::PlannerStats;
use crate::chunk_plan::GroupedChunkPlan;
use crate::chunk_plan::compile_expr_to_grouped_chunk_plan_unified;
use crate::chunk_plan::compile_node_lazy;
use crate::chunk_plan::selection_to_grouped_chunk_plan_unified_from_meta;
use crate::chunk_plan::{
    MergedCache, collect_requests_with_meta,
    materialize,
};
use crate::meta::{ZarrDatasetMeta, ZarrMeta};
use crate::{IStr, chunk_plan::*};
use std::cmp::Ordering as Ord;

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
        + HasStore,
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
        Ok(compile_expr_to_grouped_chunk_plan_unified(
            expr,
            &meta,
            self.store().clone(),
        )?)
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
        // Compile to lazy selection
        let meta = self.metadata().await?;
        let legacy_meta = meta.planning_meta();
        let (dims, dim_lengths) =
            compute_dims_and_lengths_unified(
                &meta,
            );
        let vars = legacy_meta.data_vars.clone();
        let mut ctx = LazyCompileCtx::new(
            &legacy_meta,
            Some(&meta),
            &dims,
            &vars,
        );
        let lazy_selection =
            compile_node_lazy(expr, &mut ctx)?;

        let (requests, immediate_cache) =
            collect_requests_with_meta(
                &lazy_selection,
                &legacy_meta,
                &dim_lengths,
                &dims,
            );

        let resolved_cache = self
            .resolve_batch(requests, &legacy_meta)
            .await;

        let merged = MergedCache::new(
            &*resolved_cache,
            &immediate_cache,
        );
        let selection = materialize(
            &lazy_selection,
            &legacy_meta,
            &merged,
        )
        .map_err(|e| {
            CompileError::Unsupported(format!(
                "materialization failed: {}",
                e
            ))
        })?;

        let stats =
            PlannerStats { coord_reads: 0 };

        // Convert selection to grouped chunk plan
        let grouped_plan =
            selection_to_grouped_chunk_plan_unified_from_meta(
                &selection, &meta,
            )?;

        Ok((grouped_plan, stats))
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
        legacy_meta: &ZarrDatasetMeta,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        let mut cache = HashMapCache::new();

        // Group requests by dimension
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

        // Resolve each dimension
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
        Option<IndexRange>,
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
    ) -> Option<IndexRange>;
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
        Option<IndexRange>,
    )> {
        // Get coordinate array metadata
        let coord_meta = match self
            .metadata()
            .await
            .ok()
            .and_then(|meta| {
                meta.clone()
                    .array_by_path(dim.as_ref())
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
                .map(|(req, _)| {
                    (
                        req,
                        Some(IndexRange {
                            start: 0,
                            end_exclusive: 0,
                        }),
                    )
                })
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

        // Verify with sample points
        // Note: samples must be sorted in ascending order!
        let mut samples = [
            0u64,
            chunk_size
                .saturating_sub(1)
                .min(n - 1),
            chunk_size.min(n - 1),
            (n / 2).min(n - 1),
            n - 1,
        ];
        samples.sort();

        let mut prev: Option<CoordScalar> = None;
        for &i in &samples {
            let v = self
                .scalar_at(
                    dim, i, n, chunk_size,
                    time_enc,
                )
                .await?;
            if let Some(p) = &prev {
                let ord = p.partial_cmp(&v);
                let ok = match (dir, ord) {
                    (
                        Ord::Less,
                        Some(
                            Ord::Less
                            | Ord::Equal,
                        ),
                    ) => true,
                    (
                        Ord::Greater,
                        Some(
                            Ord::Greater
                            | Ord::Equal,
                        ),
                    ) => true,
                    _ => false,
                };
                if !ok {
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

            let go_left = match (dir, strict, cmp)
            {
                (
                    Ord::Less | Ord::Equal,
                    false,
                    Some(
                        Ord::Greater | Ord::Equal,
                    ),
                ) => true,
                (
                    Ord::Less,
                    true,
                    Some(Ord::Greater),
                ) => true,
                (
                    Ord::Greater | Ord::Equal,
                    false,
                    Some(Ord::Less | Ord::Equal),
                ) => true,
                (
                    Ord::Greater,
                    true,
                    Some(Ord::Less),
                ) => true,
                _ => false,
            };

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
                match (dir, strict, cmp) {
                    (
                        Ord::Less,
                        false,
                        Some(
                            Ord::Less
                            | Ord::Equal,
                        ),
                    ) => true,
                    (
                        Ord::Less,
                        true,
                        Some(Ord::Less),
                    ) => true,
                    (
                        Ord::Greater,
                        false,
                        Some(
                            Ord::Greater
                            | Ord::Equal,
                        ),
                    ) => true,
                    (
                        Ord::Greater,
                        true,
                        Some(Ord::Greater),
                    ) => true,
                    _ => false,
                };

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
    ) -> Option<IndexRange> {
        if vr.empty {
            return Some(IndexRange {
                start: 0,
                end_exclusive: 0,
            });
        }

        // Equality case
        if let Some(eq) = &vr.eq {
            let start = self
                .lower_bound(
                    dim, eq, false, dir, n,
                    chunk_size, time_enc,
                )
                .await?;
            let end = self
                .upper_bound(
                    dim, eq, false, dir, n,
                    chunk_size, time_enc,
                )
                .await?;
            return Some(IndexRange {
                start,
                end_exclusive: end,
            });
        }

        let start = if let Some((v, bk)) = &vr.min
        {
            let strict =
                *bk == BoundKind::Exclusive;
            self.lower_bound(
                dim, v, strict, dir, n,
                chunk_size, time_enc,
            )
            .await?
        } else {
            0
        };

        let end_exclusive =
            if let Some((v, bk)) = &vr.max {
                let strict =
                    *bk == BoundKind::Exclusive;
                self.upper_bound(
                    dim, v, strict, dir, n,
                    chunk_size, time_enc,
                )
                .await?
            } else {
                n
            };

        Some(IndexRange {
            start,
            end_exclusive,
        })
    }
}
