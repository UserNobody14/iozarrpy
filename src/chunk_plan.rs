use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polars::prelude::{
    AnyValue, BooleanFunction, Expr, FunctionExpr, LiteralValue, Operator, Scalar,
};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::zarr_meta::{TimeEncoding, ZarrDatasetMeta};

#[derive(Debug, Clone)]
pub(crate) struct ChunkId {
    pub(crate) indices: Vec<u64>,
    pub(crate) origin: Vec<u64>,
    pub(crate) shape: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundKind {
    Inclusive,
    Exclusive,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CoordScalar {
    I64(i64),
    U64(u64),
    F64(f64),
    /// Nanoseconds since unix epoch.
    DatetimeNs(i64),
    /// Nanoseconds duration.
    DurationNs(i64),
    // Reserved for future: String/Binary/Categorical/etc.
}

impl CoordScalar {
    fn as_i128_orderable(&self) -> Option<i128> {
        match self {
            CoordScalar::I64(v) => Some(*v as i128),
            CoordScalar::U64(v) => Some(*v as i128),
            CoordScalar::DatetimeNs(v) => Some(*v as i128),
            CoordScalar::DurationNs(v) => Some(*v as i128),
            CoordScalar::F64(_) => None,
        }
    }

    fn partial_cmp(&self, other: &CoordScalar) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (CoordScalar::F64(a), CoordScalar::F64(b)) => a.partial_cmp(b),
            (CoordScalar::F64(a), b) => Some((*a).partial_cmp(&(b.as_i128_orderable()? as f64))?),
            (a, CoordScalar::F64(b)) => Some((a.as_i128_orderable()? as f64).partial_cmp(b)?),
            _ => Some(self.as_i128_orderable()?.cmp(&other.as_i128_orderable()?)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ValueRange {
    pub(crate) min: Option<(CoordScalar, BoundKind)>,
    pub(crate) max: Option<(CoordScalar, BoundKind)>,
    pub(crate) eq: Option<CoordScalar>,
}

impl ValueRange {
    pub(crate) fn intersect(&self, other: &ValueRange) -> ValueRange {
        let mut out = ValueRange::default();
        out.eq = match (&self.eq, &other.eq) {
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            (Some(_), Some(_)) => None, // contradictory; will become empty downstream
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        out.min = pick_tighter_min(self.min.clone(), other.min.clone());
        out.max = pick_tighter_max(self.max.clone(), other.max.clone());
        out
    }
}

fn pick_tighter_min(
    a: Option<(CoordScalar, BoundKind)>,
    b: Option<(CoordScalar, BoundKind)>,
) -> Option<(CoordScalar, BoundKind)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some((av, ak)), Some((bv, bk))) => match av.partial_cmp(&bv) {
            Some(std::cmp::Ordering::Less) => Some((bv, bk)),
            Some(std::cmp::Ordering::Greater) => Some((av, ak)),
            Some(std::cmp::Ordering::Equal) => Some((
                av,
                if ak == BoundKind::Exclusive || bk == BoundKind::Exclusive {
                    BoundKind::Exclusive
                } else {
                    BoundKind::Inclusive
                },
            )),
            None => None,
        },
    }
}

fn pick_tighter_max(
    a: Option<(CoordScalar, BoundKind)>,
    b: Option<(CoordScalar, BoundKind)>,
) -> Option<(CoordScalar, BoundKind)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some((av, ak)), Some((bv, bk))) => match av.partial_cmp(&bv) {
            Some(std::cmp::Ordering::Less) => Some((av, ak)),
            Some(std::cmp::Ordering::Greater) => Some((bv, bk)),
            Some(std::cmp::Ordering::Equal) => Some((
                av,
                if ak == BoundKind::Exclusive || bk == BoundKind::Exclusive {
                    BoundKind::Exclusive
                } else {
                    BoundKind::Inclusive
                },
            )),
            None => None,
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IndexRange {
    pub(crate) start: u64,
    pub(crate) end_exclusive: u64,
}

impl IndexRange {
    fn is_empty(&self) -> bool {
        self.end_exclusive <= self.start
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DimChunkRange {
    pub(crate) start_chunk: u64,
    pub(crate) end_chunk_inclusive: u64,
}

impl DimChunkRange {
    fn intersect(&self, other: &DimChunkRange) -> Option<DimChunkRange> {
        let s = self.start_chunk.max(other.start_chunk);
        let e = self.end_chunk_inclusive.min(other.end_chunk_inclusive);
        if e < s {
            None
        } else {
            Some(DimChunkRange {
                start_chunk: s,
                end_chunk_inclusive: e,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ChunkPlanNode {
    Empty,
    AllChunks,
    Rect(Vec<DimChunkRange>),
    Union(Vec<ChunkPlanNode>),
    PointSet(Vec<Vec<u64>>),
}

#[derive(Debug, Clone)]
pub(crate) struct ChunkPlan {
    dims: Vec<String>,
    grid_shape: Vec<u64>,
    regular_chunk_shape: Vec<u64>,
    root: ChunkPlanNode,
}

impl ChunkPlan {
    pub(crate) fn all(
        dims: Vec<String>,
        grid_shape: Vec<u64>,
        regular_chunk_shape: Vec<u64>,
    ) -> Self {
        Self {
            dims,
            grid_shape,
            regular_chunk_shape,
            root: ChunkPlanNode::AllChunks,
        }
    }

    pub(crate) fn empty(
        dims: Vec<String>,
        grid_shape: Vec<u64>,
        regular_chunk_shape: Vec<u64>,
    ) -> Self {
        Self {
            dims,
            grid_shape,
            regular_chunk_shape,
            root: ChunkPlanNode::Empty,
        }
    }

    pub(crate) fn from_root(
        dims: Vec<String>,
        grid_shape: Vec<u64>,
        regular_chunk_shape: Vec<u64>,
        root: ChunkPlanNode,
    ) -> Self {
        Self {
            dims,
            grid_shape,
            regular_chunk_shape,
            root,
        }
    }

    pub(crate) fn dims(&self) -> &[String] {
        &self.dims
    }

    pub(crate) fn grid_shape(&self) -> &[u64] {
        &self.grid_shape
    }

    pub(crate) fn regular_chunk_shape(&self) -> &[u64] {
        &self.regular_chunk_shape
    }

    pub(crate) fn into_index_iter(self) -> ChunkIndexIter {
        ChunkIndexIter::new(self.root, self.grid_shape)
    }
}

pub(crate) struct ChunkIndexIter {
    stack: Vec<OwnedIterFrame>,
}

enum OwnedIterFrame {
    Empty,
    AllChunks(AllChunksIter),
    Rect(RectIter),
    PointSet { points: Vec<Vec<u64>>, idx: usize },
    Union(UnionOwnedIter),
}

impl ChunkIndexIter {
    fn new(root: ChunkPlanNode, grid_shape: Vec<u64>) -> Self {
        let mut it = Self { stack: Vec::new() };
        it.push_node(root, grid_shape);
        it
    }

    fn push_node(&mut self, node: ChunkPlanNode, grid_shape: Vec<u64>) {
        match node {
            ChunkPlanNode::Empty => self.stack.push(OwnedIterFrame::Empty),
            ChunkPlanNode::AllChunks => self
                .stack
                .push(OwnedIterFrame::AllChunks(AllChunksIter::new(&grid_shape))),
            ChunkPlanNode::Rect(ranges) => self
                .stack
                .push(OwnedIterFrame::Rect(RectIter::new(&ranges))),
            ChunkPlanNode::PointSet(points) => {
                self.stack.push(OwnedIterFrame::PointSet { points, idx: 0 })
            }
            ChunkPlanNode::Union(children) => self.stack.push(OwnedIterFrame::Union(
                UnionOwnedIter::new(children, grid_shape),
            )),
        }
    }
}

impl Iterator for ChunkIndexIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.stack.pop()?;
            match frame {
                OwnedIterFrame::Empty => continue,
                OwnedIterFrame::AllChunks(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::AllChunks(it));
                        return Some(v);
                    }
                }
                OwnedIterFrame::Rect(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::Rect(it));
                        return Some(v);
                    }
                }
                OwnedIterFrame::PointSet { points, idx } => {
                    if idx < points.len() {
                        let v = points[idx].clone();
                        self.stack.push(OwnedIterFrame::PointSet {
                            points,
                            idx: idx + 1,
                        });
                        return Some(v);
                    }
                }
                OwnedIterFrame::Union(mut it) => {
                    if let Some(v) = it.next() {
                        self.stack.push(OwnedIterFrame::Union(it));
                        return Some(v);
                    }
                }
            }
        }
    }
}

struct AllChunksIter {
    grid_shape: Vec<u64>,
    cur: Vec<u64>,
    started: bool,
    done: bool,
}

impl AllChunksIter {
    fn new(grid_shape: &[u64]) -> Self {
        let cur = vec![0; grid_shape.len()];
        Self {
            grid_shape: grid_shape.to_vec(),
            cur,
            started: false,
            done: grid_shape.is_empty(),
        }
    }
}

impl Iterator for AllChunksIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.cur.clone());
        }
        // advance lexicographically (last dim fastest)
        for i in (0..self.cur.len()).rev() {
            self.cur[i] += 1;
            if self.cur[i] < self.grid_shape[i] {
                return Some(self.cur.clone());
            }
            self.cur[i] = 0;
        }
        self.done = true;
        None
    }
}

struct RectIter {
    ranges: Vec<DimChunkRange>,
    cur: Vec<u64>,
    started: bool,
    done: bool,
}

impl RectIter {
    fn new(ranges: &[DimChunkRange]) -> Self {
        let cur = ranges.iter().map(|r| r.start_chunk).collect::<Vec<_>>();
        let done =
            ranges.is_empty() || ranges.iter().any(|r| r.end_chunk_inclusive < r.start_chunk);
        Self {
            ranges: ranges.to_vec(),
            cur,
            started: false,
            done,
        }
    }
}

impl Iterator for RectIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.cur.clone());
        }
        for i in (0..self.cur.len()).rev() {
            self.cur[i] += 1;
            if self.cur[i] <= self.ranges[i].end_chunk_inclusive {
                return Some(self.cur.clone());
            }
            self.cur[i] = self.ranges[i].start_chunk;
        }
        self.done = true;
        None
    }
}

struct UnionOwnedIter {
    children: Vec<ChunkPlanNode>,
    child_idx: usize,
    child_iter: Option<ChunkIndexIter>,
    grid_shape: Vec<u64>,
}

impl UnionOwnedIter {
    fn new(children: Vec<ChunkPlanNode>, grid_shape: Vec<u64>) -> Self {
        Self {
            children,
            child_idx: 0,
            child_iter: None,
            grid_shape,
        }
    }
}

impl Iterator for UnionOwnedIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(it) = &mut self.child_iter {
                if let Some(v) = it.next() {
                    return Some(v);
                }
            }
            if self.child_idx >= self.children.len() {
                return None;
            }
            let node = self.children[self.child_idx].clone();
            self.child_iter = Some(ChunkIndexIter::new(node, self.grid_shape.clone()));
            self.child_idx += 1;
        }
    }
}

#[derive(Debug)]
pub(crate) enum CompileError {
    Unsupported,
    MissingPrimaryDims,
}

#[derive(Debug)]
pub(crate) enum ResolveError {
    UnsupportedCoordDtype(String),
    MissingCoord(String),
    OutOfBounds,
    Zarr(String),
}

pub(crate) trait CoordIndexResolver {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError>;

    fn index_for_value(
        &mut self,
        dim: &str,
        value: &CoordScalar,
    ) -> Result<Option<u64>, ResolveError>;

    fn coord_read_count(&self) -> u64 {
        0
    }
}

pub(crate) struct IdentityIndexResolver<'a> {
    meta: &'a ZarrDatasetMeta,
}

impl<'a> IdentityIndexResolver<'a> {
    pub(crate) fn new(meta: &'a ZarrDatasetMeta) -> Self {
        Self { meta }
    }
}

impl CoordIndexResolver for IdentityIndexResolver<'_> {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        _range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError> {
        // Only supports index-like dims where coord array doesn't exist.
        if self.meta.arrays.contains_key(dim) {
            return Ok(None);
        }
        // Without an explicit shape for a pure index dim, we can't resolve in index space yet.
        Ok(None)
    }

    fn index_for_value(
        &mut self,
        _dim: &str,
        _value: &CoordScalar,
    ) -> Result<Option<u64>, ResolveError> {
        Ok(None)
    }
}

pub(crate) struct MonotonicCoordResolver<'a> {
    meta: &'a ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    coord_arrays:
        BTreeMap<String, Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>,
    read_count: Arc<AtomicU64>,
    monotonic_cache: BTreeMap<String, Option<MonotonicDirection>>,
}

#[derive(Debug, Clone, Copy)]
enum MonotonicDirection {
    Increasing,
    Decreasing,
}

impl<'a> MonotonicCoordResolver<'a> {
    pub(crate) fn new(
        meta: &'a ZarrDatasetMeta,
        store: zarrs::storage::ReadableWritableListableStorage,
    ) -> Self {
        Self {
            meta,
            store,
            coord_arrays: BTreeMap::new(),
            read_count: Arc::new(AtomicU64::new(0)),
            monotonic_cache: BTreeMap::new(),
        }
    }

    pub(crate) fn read_count_handle(&self) -> Arc<AtomicU64> {
        self.read_count.clone()
    }

    fn coord_array(
        &mut self,
        dim: &str,
    ) -> Result<&Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>, ResolveError>
    {
        use std::collections::btree_map::Entry;

        let Some(m) = self.meta.arrays.get(dim) else {
            return Err(ResolveError::MissingCoord(dim.to_string()));
        };

        match self.coord_arrays.entry(dim.to_string()) {
            Entry::Occupied(o) => Ok(&*o.into_mut()),
            Entry::Vacant(v) => {
                let arr = Array::open(self.store.clone(), &m.path)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?;
                Ok(&*v.insert(arr))
            }
        }
    }

    fn scalar_at(&mut self, dim: &str, idx: u64) -> Result<CoordScalar, ResolveError> {
        let meta = self
            .meta
            .arrays
            .get(dim)
            .ok_or_else(|| ResolveError::MissingCoord(dim.to_string()))?;
        if meta.shape.len() != 1 {
            return Ok(CoordScalar::F64(f64::NAN)); // unsupported for now, will fail comparisons
        }
        if idx >= meta.shape[0] {
            return Err(ResolveError::OutOfBounds);
        }

        self.read_count.fetch_add(1, Ordering::Relaxed);
        let arr = self.coord_array(dim)?;
        let subset = ArraySubset::new_with_ranges(&[idx..(idx + 1)]);
        let id = arr.data_type().identifier();

        let te = meta.time_encoding.as_ref();
        match id {
            "float64" => {
                let v = arr
                    .retrieve_array_subset::<Vec<f64>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0];
                Ok(CoordScalar::F64(v))
            }
            "float32" => {
                let v = arr
                    .retrieve_array_subset::<Vec<f32>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as f64;
                Ok(CoordScalar::F64(v))
            }
            "int64" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i64>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0];
                Ok(apply_time_encoding(raw, te))
            }
            "int32" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i32>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as i64;
                Ok(apply_time_encoding(raw, te))
            }
            "int16" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i16>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as i64;
                Ok(apply_time_encoding(raw, te))
            }
            "int8" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i8>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as i64;
                Ok(apply_time_encoding(raw, te))
            }
            "uint64" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u64>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0];
                Ok(CoordScalar::U64(v))
            }
            "uint32" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u32>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as u64;
                Ok(CoordScalar::U64(v))
            }
            "uint16" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u16>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as u64;
                Ok(CoordScalar::U64(v))
            }
            "uint8" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u8>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as u64;
                Ok(CoordScalar::U64(v))
            }
            other => Err(ResolveError::UnsupportedCoordDtype(other.to_string())),
        }
    }

    fn ensure_monotonic(&mut self, dim: &str) -> Result<Option<MonotonicDirection>, ResolveError> {
        if let Some(cached) = self.monotonic_cache.get(dim) {
            return Ok(*cached);
        }
        let Some(meta) = self.meta.arrays.get(dim) else {
            self.monotonic_cache.insert(dim.to_string(), None);
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            self.monotonic_cache.insert(dim.to_string(), None);
            return Ok(None);
        }
        let n = meta.shape[0];
        if n < 2 {
            self.monotonic_cache
                .insert(dim.to_string(), Some(MonotonicDirection::Increasing));
            return Ok(Some(MonotonicDirection::Increasing));
        }

        // Cheap monotonic heuristic: sample a few points, including around a chunk boundary.
        // This is conservative: if we can't confirm, we return None (no binary-search pruning).
        let first = self.scalar_at(dim, 0)?;
        let last = self.scalar_at(dim, n - 1)?;
        let dir = match first.partial_cmp(&last) {
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => {
                MonotonicDirection::Increasing
            }
            Some(std::cmp::Ordering::Greater) => MonotonicDirection::Decreasing,
            None => {
                self.monotonic_cache.insert(dim.to_string(), None);
                return Ok(None);
            }
        };

        let arr = self.coord_array(dim)?;
        let reg_chunk = arr
            .chunk_shape(&[0])
            .map_err(|e| ResolveError::Zarr(e.to_string()))?
            .get(0)
            .map(|nz| nz.get())
            .unwrap_or(1);
        let sample_idxs = [
            0u64,
            (reg_chunk.saturating_sub(1)).min(n - 1),
            (reg_chunk).min(n - 1),
            (n / 2).min(n - 1),
            n - 1,
        ];

        let mut prev: Option<CoordScalar> = None;
        for &i in &sample_idxs {
            let v = self.scalar_at(dim, i)?;
            if let Some(p) = &prev {
                let ord = p.partial_cmp(&v);
                let ok = match (dir, ord) {
                    (
                        MonotonicDirection::Increasing,
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                    ) => true,
                    (
                        MonotonicDirection::Decreasing,
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                    ) => true,
                    _ => false,
                };
                if !ok {
                    self.monotonic_cache.insert(dim.to_string(), None);
                    return Ok(None);
                }
            }
            prev = Some(v);
        }

        self.monotonic_cache.insert(dim.to_string(), Some(dir));
        Ok(Some(dir))
    }

    fn lower_bound(
        &mut self,
        dim: &str,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDirection,
        n: u64,
    ) -> Result<u64, ResolveError> {
        // For increasing: first idx with value > target (strict) or >= target (!strict).
        // For decreasing: first idx with value < target (strict) or <= target (!strict).
        let mut lo = 0u64;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at(dim, mid)?;
            let cmp = v.partial_cmp(target);
            let go_left = match (dir, strict, cmp) {
                (
                    MonotonicDirection::Increasing,
                    false,
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDirection::Increasing, true, Some(std::cmp::Ordering::Greater)) => true,
                (
                    MonotonicDirection::Decreasing,
                    false,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDirection::Decreasing, true, Some(std::cmp::Ordering::Less)) => true,
                _ => false,
            };
            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    fn upper_bound(
        &mut self,
        dim: &str,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDirection,
        n: u64,
    ) -> Result<u64, ResolveError> {
        // Return end_exclusive for max bound:
        // For increasing: first idx with value >= target (strict) or > target (!strict)??? Wait:
        // We want end_exclusive such that values satisfy value < max (Exclusive) or <= max (Inclusive).
        // So compute first idx that violates that, i.e. value >= max (Exclusive) or value > max (Inclusive).
        let mut lo = 0u64;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at(dim, mid)?;
            let cmp = v.partial_cmp(target);
            let go_left = match (dir, strict, cmp) {
                (
                    MonotonicDirection::Increasing,
                    true,
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                ) => true, // >= max
                (MonotonicDirection::Increasing, false, Some(std::cmp::Ordering::Greater)) => true, // > max
                (
                    MonotonicDirection::Decreasing,
                    true,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                ) => true, // <= max violates for decreasing? symmetric
                (MonotonicDirection::Decreasing, false, Some(std::cmp::Ordering::Less)) => true,
                _ => false,
            };
            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }
}

impl CoordIndexResolver for MonotonicCoordResolver<'_> {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError> {
        let Some(meta) = self.meta.arrays.get(dim) else {
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            return Ok(None);
        }
        let n = meta.shape[0];
        if n == 0 {
            return Ok(Some(IndexRange {
                start: 0,
                end_exclusive: 0,
            }));
        }

        let Some(dir) = self.ensure_monotonic(dim)? else {
            return Ok(None);
        };

        // Equality is treated as a tiny closed range in index space using two bounds.
        if let Some(eq) = &range.eq {
            let start = self.lower_bound(dim, eq, false, dir, n)?;
            let end_excl = self.upper_bound(dim, eq, false, dir, n)?;
            let out = IndexRange {
                start,
                end_exclusive: end_excl,
            };
            return Ok(Some(out));
        }

        let start = if let Some((v, bk)) = &range.min {
            let strict = *bk == BoundKind::Exclusive;
            self.lower_bound(dim, v, strict, dir, n)?
        } else {
            0
        };
        let end_exclusive = if let Some((v, bk)) = &range.max {
            let strict = *bk == BoundKind::Exclusive;
            // strict means "< v" so violation at >= v; non-strict means "<= v" so violation at > v.
            self.upper_bound(dim, v, strict, dir, n)?
        } else {
            n
        };

        Ok(Some(IndexRange {
            start,
            end_exclusive,
        }))
    }

    fn index_for_value(
        &mut self,
        dim: &str,
        value: &CoordScalar,
    ) -> Result<Option<u64>, ResolveError> {
        let Some(meta) = self.meta.arrays.get(dim) else {
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            return Ok(None);
        }
        let n = meta.shape[0];
        let Some(dir) = self.ensure_monotonic(dim)? else {
            return Ok(None);
        };
        let start = self.lower_bound(dim, value, false, dir, n)?;
        let end = self.upper_bound(dim, value, false, dir, n)?;
        if start < end {
            Ok(Some(start))
        } else {
            Ok(None)
        }
    }

    fn coord_read_count(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }
}

fn apply_time_encoding(raw: i64, te: Option<&TimeEncoding>) -> CoordScalar {
    if let Some(enc) = te {
        let ns = enc.decode(raw);
        if enc.is_duration {
            CoordScalar::DurationNs(ns)
        } else {
            CoordScalar::DatetimeNs(ns)
        }
    } else {
        CoordScalar::I64(raw)
    }
}

fn literal_anyvalue(lit: &LiteralValue) -> Option<AnyValue<'static>> {
    match lit {
        LiteralValue::Scalar(s) => Some(s.clone().into_value().into_static()),
        LiteralValue::Dyn(d) => {
            // Polars (via pyo3-polars) commonly serializes Python literals as dyn literals
            // (e.g. "dyn int: 20"). Polars doesn't currently expose a stable public API
            // to convert `DynLiteralValue` -> `AnyValue` in our dependency surface, so
            // we conservatively parse the debug representation for the primitive types
            // we need for chunk planning.
            //
            // If parsing fails, we return None and planning will fall back to AllChunks.
            let s = format!("{d:?}");
            let s = s.trim();
            if let Some(rest) = s.strip_prefix("dyn int:") {
                let v = rest.trim().parse::<i64>().ok()?;
                return Some(AnyValue::Int64(v).into_static());
            }
            if let Some(rest) = s.strip_prefix("dyn float:") {
                let v = rest.trim().parse::<f64>().ok()?;
                return Some(AnyValue::Float64(v).into_static());
            }
            if let Some(rest) = s.strip_prefix("dyn bool:") {
                let v = rest.trim().parse::<bool>().ok()?;
                return Some(AnyValue::Boolean(v).into_static());
            }
            None
        }
        _ => None,
    }
}

fn literal_to_scalar(
    lit: &LiteralValue,
    time_encoding: Option<&TimeEncoding>,
) -> Option<CoordScalar> {
    let _ = time_encoding;
    match literal_anyvalue(lit)? {
        AnyValue::Int64(v) => Some(CoordScalar::I64(v)),
        AnyValue::Int32(v) => Some(CoordScalar::I64(v as i64)),
        AnyValue::Int16(v) => Some(CoordScalar::I64(v as i64)),
        AnyValue::Int8(v) => Some(CoordScalar::I64(v as i64)),
        AnyValue::UInt64(v) => Some(CoordScalar::U64(v)),
        AnyValue::UInt32(v) => Some(CoordScalar::U64(v as u64)),
        AnyValue::UInt16(v) => Some(CoordScalar::U64(v as u64)),
        AnyValue::UInt8(v) => Some(CoordScalar::U64(v as u64)),
        AnyValue::Float64(v) => Some(CoordScalar::F64(v)),
        AnyValue::Float32(v) => Some(CoordScalar::F64(v as f64)),
        AnyValue::Datetime(value, time_unit, _) => {
            let ns = match time_unit {
                polars::prelude::TimeUnit::Nanoseconds => value,
                polars::prelude::TimeUnit::Microseconds => value * 1_000,
                polars::prelude::TimeUnit::Milliseconds => value * 1_000_000,
            };
            let _ = time_encoding;
            Some(CoordScalar::DatetimeNs(ns))
        }
        AnyValue::Date(days) => {
            let ns = days as i64 * 86400 * 1_000_000_000;
            let _ = time_encoding;
            Some(CoordScalar::DatetimeNs(ns))
        }
        AnyValue::Duration(value, time_unit) => {
            let ns = match time_unit {
                polars::prelude::TimeUnit::Nanoseconds => value,
                polars::prelude::TimeUnit::Microseconds => value * 1_000,
                polars::prelude::TimeUnit::Milliseconds => value * 1_000_000,
            };
            let _ = time_encoding;
            Some(CoordScalar::DurationNs(ns))
        }
        _ => None,
    }
}

fn col_lit(col_side: &Expr, lit_side: &Expr) -> Option<(String, LiteralValue)> {
    let col_side = strip_wrappers(col_side);
    let lit_side = strip_wrappers(lit_side);
    match (col_side, lit_side) {
        (Expr::Column(name), Expr::Literal(lit)) => Some((name.to_string(), lit.clone())),
        _ => None,
    }
}

fn strip_wrappers(mut e: &Expr) -> &Expr {
    loop {
        match e {
            Expr::Alias(inner, _) => e = inner.as_ref(),
            Expr::Cast { expr, .. } => e = expr.as_ref(),
            _ => return e,
        }
    }
}

fn reverse_operator(op: Operator) -> Operator {
    match op {
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        _ => op,
    }
}

fn chunk_ranges_for_index_range(
    idx: IndexRange,
    chunk_size: u64,
    grid_dim: u64,
) -> Option<DimChunkRange> {
    if idx.is_empty() {
        return None;
    }
    let chunk_start = idx.start / chunk_size;
    let last = idx.end_exclusive.saturating_sub(1);
    let chunk_end = last / chunk_size;
    if chunk_start >= grid_dim {
        return None;
    }
    let end = chunk_end.min(grid_dim.saturating_sub(1));
    Some(DimChunkRange {
        start_chunk: chunk_start,
        end_chunk_inclusive: end,
    })
}

fn rect_all_dims(grid_shape: &[u64]) -> Vec<DimChunkRange> {
    grid_shape
        .iter()
        .map(|&g| DimChunkRange {
            start_chunk: 0,
            end_chunk_inclusive: g.saturating_sub(1),
        })
        .collect()
}

fn and_nodes(a: ChunkPlanNode, b: ChunkPlanNode) -> ChunkPlanNode {
    match (a, b) {
        (ChunkPlanNode::Empty, _) | (_, ChunkPlanNode::Empty) => ChunkPlanNode::Empty,
        (ChunkPlanNode::AllChunks, x) | (x, ChunkPlanNode::AllChunks) => x,
        (ChunkPlanNode::Union(xs), y) => {
            ChunkPlanNode::Union(xs.into_iter().map(|x| and_nodes(x, y.clone())).collect())
        }
        (x, ChunkPlanNode::Union(ys)) => {
            ChunkPlanNode::Union(ys.into_iter().map(|y| and_nodes(x.clone(), y)).collect())
        }
        (ChunkPlanNode::Rect(a), ChunkPlanNode::Rect(b)) => {
            if a.len() != b.len() {
                return ChunkPlanNode::Empty;
            }
            let mut out = Vec::with_capacity(a.len());
            for i in 0..a.len() {
                let Some(r) = a[i].intersect(&b[i]) else {
                    return ChunkPlanNode::Empty;
                };
                out.push(r);
            }
            ChunkPlanNode::Rect(out)
        }
        // Anything else: fall back to a conservative union of both (still a superset).
        // This avoids accidentally dropping candidates.
        (x, y) => ChunkPlanNode::Union(vec![x, y]),
    }
}

fn or_nodes(a: ChunkPlanNode, b: ChunkPlanNode) -> ChunkPlanNode {
    match (a, b) {
        (ChunkPlanNode::Empty, x) | (x, ChunkPlanNode::Empty) => x,
        (ChunkPlanNode::AllChunks, _) | (_, ChunkPlanNode::AllChunks) => ChunkPlanNode::AllChunks,
        (ChunkPlanNode::Union(mut xs), ChunkPlanNode::Union(ys)) => {
            xs.extend(ys);
            ChunkPlanNode::Union(xs)
        }
        (ChunkPlanNode::Union(mut xs), y) => {
            xs.push(y);
            ChunkPlanNode::Union(xs)
        }
        (x, ChunkPlanNode::Union(mut ys)) => {
            ys.insert(0, x);
            ChunkPlanNode::Union(ys)
        }
        (x, y) => ChunkPlanNode::Union(vec![x, y]),
    }
}

pub(crate) struct PlannerStats {
    pub(crate) coord_reads: u64,
}

pub(crate) fn compile_expr_to_chunk_plan(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    primary_var: &str,
) -> Result<(ChunkPlan, PlannerStats), CompileError> {
    let Some(primary_meta) = meta.arrays.get(primary_var) else {
        return Err(CompileError::MissingPrimaryDims);
    };
    let dims = if !primary_meta.dims.is_empty() {
        primary_meta.dims.clone()
    } else {
        meta.dims.clone()
    };

    let primary =
        Array::open(store.clone(), &primary_meta.path).map_err(|_| CompileError::Unsupported)?;
    let grid_shape = primary.chunk_grid().grid_shape().to_vec();
    let zero = vec![0u64; primary.dimensionality()];
    let chunk_shape_nz = primary
        .chunk_shape(&zero)
        .map_err(|_| CompileError::Unsupported)?;
    let regular_chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

    let mut resolver = MonotonicCoordResolver::new(meta, store);
    let root = compile_node(
        expr,
        meta,
        &dims,
        &grid_shape,
        &regular_chunk_shape,
        &mut resolver,
    )
    .unwrap_or(ChunkPlanNode::AllChunks);
    let grid_shape_vec = grid_shape.to_vec();
    let plan = ChunkPlan::from_root(dims, grid_shape_vec, regular_chunk_shape, root);
    let stats = PlannerStats {
        coord_reads: resolver.coord_read_count(),
    };
    Ok((plan, stats))
}

fn compile_node(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    match expr {
        Expr::Alias(inner, _) => {
            compile_node(inner, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::KeepName(inner) => {
            compile_node(inner, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::RenameAlias { expr, .. } => {
            compile_node(expr, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::Cast { expr, .. } => {
            compile_node(expr, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::Sort { expr, .. } => {
            compile_node(expr, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::SortBy { expr, .. } => {
            compile_node(expr, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::Explode { input, .. } => {
            compile_node(input, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::Slice { input, .. } => {
            compile_node(input, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        // For window expressions, just compile the function expression only for now.
        // TODO: handle partition_by and order_by if needed.
        Expr::Over { function, .. } => compile_node(
            function,
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        // Expr::Rolling {
        //     function,
        //     index_column,
        //     period,
        //     offset,
        //     closed_window,
        // } => compile_node(
        //     function,
        //     meta,
        //     dims,
        //     grid_shape,
        //     regular_chunk_shape,
        //     resolver,
        // ),
        // Expr::Window { function, .. } => compile_node(function, meta, dims, grid_shape, regular_chunk_shape, resolver),
        // If a filter expression is used where we expect a predicate, focus on the predicate.
        Expr::Filter { by, .. } => {
            compile_node(by, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }
        Expr::BinaryExpr { left, op, right } => {
            match op {
                Operator::And | Operator::LogicalAnd => {
                    // If one side is unsupported, keep whatever constraints we can from the other.
                    let a =
                        compile_node(left, meta, dims, grid_shape, regular_chunk_shape, resolver)
                            .unwrap_or(ChunkPlanNode::AllChunks);
                    let b =
                        compile_node(right, meta, dims, grid_shape, regular_chunk_shape, resolver)
                            .unwrap_or(ChunkPlanNode::AllChunks);
                    Ok(and_nodes(a, b))
                }
                Operator::Or | Operator::LogicalOr => {
                    // For OR, if either side is unsupported we conservatively plan all chunks.
                    let a =
                        compile_node(left, meta, dims, grid_shape, regular_chunk_shape, resolver)
                            .unwrap_or(ChunkPlanNode::AllChunks);
                    let b =
                        compile_node(right, meta, dims, grid_shape, regular_chunk_shape, resolver)
                            .unwrap_or(ChunkPlanNode::AllChunks);
                    Ok(or_nodes(a, b))
                }
                Operator::Eq | Operator::GtEq | Operator::Gt | Operator::LtEq | Operator::Lt => {
                    if let Some((col, lit)) = col_lit(left, right).or_else(|| col_lit(right, left))
                    {
                        let op_eff = if matches!(left.as_ref(), Expr::Literal(_)) {
                            reverse_operator(*op)
                        } else {
                            *op
                        };
                        compile_cmp(
                            &col,
                            op_eff,
                            &lit,
                            meta,
                            dims,
                            grid_shape,
                            regular_chunk_shape,
                            resolver,
                        )
                    } else {
                        Err(CompileError::Unsupported)
                    }
                }
                _ => Err(CompileError::Unsupported),
            }
        }
        Expr::Literal(lit) => {
            // Only boolean-ish literals can be predicates.
            match literal_anyvalue(lit) {
                Some(AnyValue::Boolean(true)) => Ok(ChunkPlanNode::AllChunks),
                Some(AnyValue::Boolean(false)) => Ok(ChunkPlanNode::Empty),
                // In Polars filtering, null predicate behaves like "keep nothing".
                Some(AnyValue::Null) => Ok(ChunkPlanNode::Empty),
                _ => Err(CompileError::Unsupported),
            }
        }
        Expr::Function { input, function } => {
            match function {
                FunctionExpr::Boolean(bf) => compile_boolean_function(
                    bf,
                    input,
                    meta,
                    dims,
                    grid_shape,
                    regular_chunk_shape,
                    resolver,
                ),
                // Most functions transform values in ways that we can't safely map to chunk-level constraints.
                _ => Err(CompileError::Unsupported),
            }
        }

        // Variants without a meaningful chunk-planning representation.
        Expr::Element
        | Expr::Column(_)
        | Expr::Selector(_)
        | Expr::DataTypeFunction(_)
        | Expr::Gather { .. }
        | Expr::Agg(_)
        | Expr::Ternary { .. }
        | Expr::Len
        | Expr::AnonymousFunction { .. }
        | Expr::Eval { .. }
        | Expr::SubPlan(_, _)
        | Expr::Field(_) => Err(CompileError::Unsupported),
    }
}

fn compile_boolean_function(
    bf: &BooleanFunction,
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    match bf {
        BooleanFunction::Not => {
            let [arg] = input else {
                return Err(CompileError::Unsupported);
            };
            // Try constant fold first.
            if let Expr::Literal(lit) = strip_wrappers(arg) {
                return match literal_anyvalue(lit) {
                    Some(AnyValue::Boolean(true)) => Ok(ChunkPlanNode::Empty),
                    Some(AnyValue::Boolean(false)) => Ok(ChunkPlanNode::AllChunks),
                    Some(AnyValue::Null) => Ok(ChunkPlanNode::Empty),
                    _ => Ok(ChunkPlanNode::AllChunks),
                };
            }

            // If the inner predicate is known to match nothing, NOT(...) matches everything.
            // Otherwise we can't represent complements with current plan nodes.
            match compile_node(arg, meta, dims, grid_shape, regular_chunk_shape, resolver)
                .unwrap_or(ChunkPlanNode::AllChunks)
            {
                ChunkPlanNode::Empty => Ok(ChunkPlanNode::AllChunks),
                _ => Ok(ChunkPlanNode::AllChunks),
            }
        }
        BooleanFunction::IsNull | BooleanFunction::IsNotNull => {
            let [arg] = input else {
                return Err(CompileError::Unsupported);
            };

            // Constant fold when possible; otherwise don't constrain.
            if let Expr::Literal(lit) = strip_wrappers(arg) {
                let is_null = matches!(literal_anyvalue(lit), Some(AnyValue::Null));
                let keep = match bf {
                    BooleanFunction::IsNull => is_null,
                    BooleanFunction::IsNotNull => !is_null,
                    _ => unreachable!(),
                };
                return Ok(if keep {
                    ChunkPlanNode::AllChunks
                } else {
                    ChunkPlanNode::Empty
                });
            }
            Ok(ChunkPlanNode::AllChunks)
        }
        _ => {
            // Future-proof handling for optional Polars boolean features without hard-referencing
            // cfg-gated variants (e.g. `is_in`, `is_between`).
            let name = bf.to_string();
            match name.as_str() {
                "is_between" => {
                    compile_is_between(input, meta, dims, grid_shape, regular_chunk_shape, resolver)
                }
                "is_in" => {
                    compile_is_in(input, meta, dims, grid_shape, regular_chunk_shape, resolver)
                }
                _ => Ok(ChunkPlanNode::AllChunks),
            }
        }
    }
}

fn expr_to_col_name(e: &Expr) -> Option<&str> {
    match strip_wrappers(e) {
        Expr::Column(name) => Some(name.as_str()),
        _ => None,
    }
}

fn compile_is_between(
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let [expr, low, high] = input else {
        return Err(CompileError::Unsupported);
    };
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ChunkPlanNode::AllChunks);
    };
    let Expr::Literal(low_lit) = strip_wrappers(low) else {
        return Ok(ChunkPlanNode::AllChunks);
    };
    let Expr::Literal(high_lit) = strip_wrappers(high) else {
        return Ok(ChunkPlanNode::AllChunks);
    };

    // Conservatively assume a closed interval (inclusive bounds) to avoid false negatives.
    let a = compile_cmp(
        col,
        Operator::GtEq,
        low_lit,
        meta,
        dims,
        grid_shape,
        regular_chunk_shape,
        resolver,
    )
    .unwrap_or(ChunkPlanNode::AllChunks);
    let b = compile_cmp(
        col,
        Operator::LtEq,
        high_lit,
        meta,
        dims,
        grid_shape,
        regular_chunk_shape,
        resolver,
    )
    .unwrap_or(ChunkPlanNode::AllChunks);
    Ok(and_nodes(a, b))
}

fn compile_is_in(
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let [expr, list] = input else {
        return Err(CompileError::Unsupported);
    };
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ChunkPlanNode::AllChunks);
    };

    let Expr::Literal(list_lit) = strip_wrappers(list) else {
        return Ok(ChunkPlanNode::AllChunks);
    };

    match list_lit {
        LiteralValue::Series(s) => {
            let series = &**s;
            // Prevent pathological unions for huge lists.
            if series.len() > 4096 {
                return Ok(ChunkPlanNode::AllChunks);
            }

            let mut out: Option<ChunkPlanNode> = None;
            for av in series.iter() {
                let av = av.into_static();
                if matches!(av, AnyValue::Null) {
                    // Null membership semantics depend on `nulls_equal`; we avoid constraining.
                    return Ok(ChunkPlanNode::AllChunks);
                }

                let lit = LiteralValue::Scalar(Scalar::new(series.dtype().clone(), av));
                let node = compile_cmp(
                    col,
                    Operator::Eq,
                    &lit,
                    meta,
                    dims,
                    grid_shape,
                    regular_chunk_shape,
                    resolver,
                )
                .unwrap_or(ChunkPlanNode::AllChunks);

                // If any element falls back to AllChunks, the whole IN predicate becomes unconstrainable.
                if matches!(node, ChunkPlanNode::AllChunks) {
                    return Ok(ChunkPlanNode::AllChunks);
                }
                out = Some(match out.take() {
                    None => node,
                    Some(acc) => or_nodes(acc, node),
                });
            }
            Ok(out.unwrap_or(ChunkPlanNode::Empty))
        }
        _ => Ok(ChunkPlanNode::AllChunks),
    }
}

fn compile_cmp(
    col: &str,
    op: Operator,
    lit: &LiteralValue,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let dim_idx = dims
        .iter()
        .position(|d| d == col)
        .ok_or(CompileError::Unsupported)?;

    let time_encoding = meta.arrays.get(col).and_then(|a| a.time_encoding.as_ref());
    let Some(scalar) = literal_to_scalar(lit, time_encoding) else {
        return Err(CompileError::Unsupported);
    };

    let mut vr = ValueRange::default();
    match op {
        Operator::Eq => vr.eq = Some(scalar),
        Operator::Gt => vr.min = Some((scalar, BoundKind::Exclusive)),
        Operator::GtEq => vr.min = Some((scalar, BoundKind::Inclusive)),
        Operator::Lt => vr.max = Some((scalar, BoundKind::Exclusive)),
        Operator::LtEq => vr.max = Some((scalar, BoundKind::Inclusive)),
        _ => return Err(CompileError::Unsupported),
    }

    let Some(idx_range) = resolver
        .index_range_for_value_range(col, &vr)
        .map_err(|_| CompileError::Unsupported)?
    else {
        return Err(CompileError::Unsupported);
    };

    if idx_range.is_empty() {
        return Ok(ChunkPlanNode::Empty);
    }
    let Some(dim_range) = chunk_ranges_for_index_range(
        idx_range,
        regular_chunk_shape[dim_idx].max(1),
        grid_shape[dim_idx],
    ) else {
        return Ok(ChunkPlanNode::Empty);
    };

    let mut rect = rect_all_dims(grid_shape);
    rect[dim_idx] = dim_range;
    Ok(ChunkPlanNode::Rect(rect))
}
