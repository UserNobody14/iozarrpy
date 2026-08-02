//! Arrow-backed column storage decoded directly from zarrs.
//!
//! `ColumnData` retains its public API (variant names, helper methods)
//! but internally stores Polars Arrow array types. Numeric variants are
//! `PrimitiveArray<T>` which wraps a refcounted `Buffer<T>`; bool is
//! `BooleanArray` (bit-packed); strings/binary are `Utf8Array<i64>` /
//! `BinaryArray<i64>` (Arrow large-offset variants). All cloning is
//! `O(1)` via Arc, and slicing is `O(1)` over the buffer length.
//!
//! Decoding happens through the [`DecodedChunk`] newtype, which
//! implements [`zarrs::array::FromArrayBytes`]. Fixed-width primitive
//! values transmute the raw bytes via `bytemuck` (zero-copy on aligned
//! platforms). Variable-length payloads convert the `usize` zarrs
//! offsets to `i64` Arrow offsets up front. Bool bytes are bit-packed
//! into an Arrow [`Bitmap`].

use std::sync::Arc;

use polars::prelude::Series;
use polars_arrow::array::{
    Array, BinaryArray, BooleanArray,
    PrimitiveArray, Utf8Array,
};
use polars_arrow::bitmap::{
    Bitmap, MutableBitmap,
};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::OffsetsBuffer;
use zarrs::array::{
    ArrayBytes, ArrayError, DataType,
    FromArrayBytes, convert_from_bytes_slice,
    transmute_from_bytes_vec,
};
use zarrs::plugin::ZarrVersion;

// ============================================================================
// repeat_tile helpers (kept verbatim — used on values buffers)
// ============================================================================

/// Build a Vec by repeating each element of `src` `inner_repeat`
/// times, then tiling the resulting pattern `tile_count` times.
///
/// Total output length = `src.len() * inner_repeat * tile_count`.
fn repeat_tile_slice<T: Copy>(
    src: &[T],
    inner_repeat: usize,
    tile_count: usize,
) -> Vec<T> {
    if src.is_empty()
        || inner_repeat == 0
        || tile_count == 0
    {
        return Vec::new();
    }

    let tile_len = src.len() * inner_repeat;
    let total = tile_len * tile_count;
    let mut output = Vec::with_capacity(total);

    if inner_repeat == 1 {
        output.extend_from_slice(src);
    } else {
        for &val in src {
            let start = output.len();
            output.push(val);
            while output.len() - start
                < inner_repeat
            {
                let filled = output.len() - start;
                let to_copy = (inner_repeat
                    - filled)
                    .min(filled);
                output.extend_from_within(
                    start..start + to_copy,
                );
            }
        }
    }

    if tile_count > 1 {
        for _ in 1..tile_count {
            output
                .extend_from_within(0..tile_len);
        }
    }

    output
}

// ============================================================================
// Arrow-backed enum
// ============================================================================

/// Owned column data backed by Polars Arrow arrays.
///
/// Variant names mirror the zarr dtype family they decode from. All
/// payloads are reference-counted internally, so `Clone` is `O(1)`
/// (refcount bump) and slicing is also `O(1)`.
#[derive(Debug, Clone)]
pub enum ColumnData {
    Bool(BooleanArray),
    I8(PrimitiveArray<i8>),
    I16(PrimitiveArray<i16>),
    I32(PrimitiveArray<i32>),
    I64(PrimitiveArray<i64>),
    U8(PrimitiveArray<u8>),
    U16(PrimitiveArray<u16>),
    U32(PrimitiveArray<u32>),
    U64(PrimitiveArray<u64>),
    F32(PrimitiveArray<f32>),
    F64(PrimitiveArray<f64>),
    Str(Utf8Array<i64>),
    Bin(BinaryArray<i64>),
}

// ============================================================================
// Constructors used by call sites that synthesize columns (not zarrs decoded)
// ============================================================================

impl ColumnData {
    /// Build an `i64` column from an owned vector of values, with no
    /// validity mask.
    pub fn from_i64_vec(
        values: Vec<i64>,
    ) -> Self {
        Self::I64(PrimitiveArray::from_vec(
            values,
        ))
    }

    /// Empty `i64` column. Used by chunk-to-df paths to short-circuit
    /// zero-length range reads.
    pub fn empty_i64() -> Self {
        Self::I64(PrimitiveArray::from_vec(
            Vec::new(),
        ))
    }
}

// ============================================================================
// Typed accessors — replace direct `ColumnData::F64(v)` Vec destructures.
// ============================================================================

impl ColumnData {
    /// Slice into the underlying `f64` values, ignoring validity. Returns
    /// `None` if the column is not `F64`.
    #[inline]
    pub fn as_f64_values(
        &self,
    ) -> Option<&[f64]> {
        match self {
            ColumnData::F64(v) => {
                Some(v.values().as_slice())
            }
            _ => None,
        }
    }

    /// Slice into the underlying `f32` values, ignoring validity. Returns
    /// `None` if the column is not `F32`.
    #[inline]
    pub fn as_f32_values(
        &self,
    ) -> Option<&[f32]> {
        match self {
            ColumnData::F32(v) => {
                Some(v.values().as_slice())
            }
            _ => None,
        }
    }

    /// Slice into the underlying `i64` values, ignoring validity. Returns
    /// `None` if the column is not `I64`.
    #[inline]
    pub fn as_i64_values(
        &self,
    ) -> Option<&[i64]> {
        match self {
            ColumnData::I64(v) => {
                Some(v.values().as_slice())
            }
            _ => None,
        }
    }
}

// ============================================================================
// Core API used across the scan pipeline
// ============================================================================

impl ColumnData {
    pub(crate) fn len(&self) -> usize {
        match self {
            ColumnData::Bool(v) => v.len(),
            ColumnData::I8(v) => v.len(),
            ColumnData::I16(v) => v.len(),
            ColumnData::I32(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::U8(v) => v.len(),
            ColumnData::U16(v) => v.len(),
            ColumnData::U32(v) => v.len(),
            ColumnData::U64(v) => v.len(),
            ColumnData::F32(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::Str(v) => v.len(),
            ColumnData::Bin(v) => v.len(),
        }
    }

    /// Zero-copy slice. `start..start+len` must be in bounds.
    pub(crate) fn slice(
        &self,
        start: usize,
        len: usize,
    ) -> ColumnData {
        match self {
            ColumnData::Bool(v) => {
                ColumnData::Bool(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::I8(v) => ColumnData::I8(
                v.clone().sliced(start, len),
            ),
            ColumnData::I16(v) => {
                ColumnData::I16(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::I64(v) => {
                ColumnData::I64(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::U8(v) => ColumnData::U8(
                v.clone().sliced(start, len),
            ),
            ColumnData::U16(v) => {
                ColumnData::U16(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::U64(v) => {
                ColumnData::U64(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::Str(v) => {
                ColumnData::Str(
                    v.clone().sliced(start, len),
                )
            }
            ColumnData::Bin(v) => {
                ColumnData::Bin(
                    v.clone().sliced(start, len),
                )
            }
        }
    }

    pub(crate) fn take_indices(
        &self,
        indices: &[usize],
    ) -> ColumnData {
        match self {
            ColumnData::Bool(v) => {
                let bm = bitmap_from_indexed(
                    v.values(),
                    indices.iter().copied(),
                );
                ColumnData::Bool(
                    BooleanArray::new(
                        ArrowDataType::Boolean,
                        bm,
                        None,
                    ),
                )
            }
            ColumnData::I8(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::I8)
                .unwrap()
            }
            ColumnData::I16(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::I16)
                .unwrap()
            }
            ColumnData::I32(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::I32)
                .unwrap()
            }
            ColumnData::I64(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::I64)
                .unwrap()
            }
            ColumnData::U8(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::U8)
                .unwrap()
            }
            ColumnData::U16(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::U16)
                .unwrap()
            }
            ColumnData::U32(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::U32)
                .unwrap()
            }
            ColumnData::U64(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::U64)
                .unwrap()
            }
            ColumnData::F32(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::F32)
                .unwrap()
            }
            ColumnData::F64(v) => {
                primitive_gather(
                    v,
                    indices.iter().copied(),
                )
                .map(ColumnData::F64)
                .unwrap()
            }
            ColumnData::Str(v) => {
                ColumnData::Str(gather_utf8(
                    v,
                    indices.iter().copied(),
                ))
            }
            ColumnData::Bin(v) => {
                ColumnData::Bin(gather_binary(
                    v,
                    indices.iter().copied(),
                ))
            }
        }
    }

    pub(crate) fn get_i64(
        &self,
        idx: usize,
    ) -> Option<i64> {
        match self {
            ColumnData::I64(v) => {
                Some(v.values()[idx])
            }
            ColumnData::I32(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::I16(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::I8(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::U64(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::U32(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::U16(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::U8(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::F32(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::F64(v) => {
                Some(v.values()[idx] as i64)
            }
            ColumnData::Bool(v) => {
                Some(i64::from(
                    v.values().get_bit(idx),
                ))
            }
            ColumnData::Str(_)
            | ColumnData::Bin(_) => None,
        }
    }

    pub(crate) fn map_i64(
        &self,
        f: impl Fn(i64) -> i64,
    ) -> ColumnData {
        match self {
            ColumnData::I64(v) => {
                primitive_map(v, &f)
                    .map(ColumnData::I64)
                    .unwrap()
            }
            ColumnData::I32(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as i32
                })
                .map(ColumnData::I32)
                .unwrap()
            }
            ColumnData::I16(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as i16
                })
                .map(ColumnData::I16)
                .unwrap()
            }
            ColumnData::I8(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as i8
                })
                .map(ColumnData::I8)
                .unwrap()
            }
            ColumnData::U64(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as u64
                })
                .map(ColumnData::U64)
                .unwrap()
            }
            ColumnData::U32(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as u32
                })
                .map(ColumnData::U32)
                .unwrap()
            }
            ColumnData::U16(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as u16
                })
                .map(ColumnData::U16)
                .unwrap()
            }
            ColumnData::U8(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as u8
                })
                .map(ColumnData::U8)
                .unwrap()
            }
            ColumnData::Bool(v) => {
                let n = v.len();
                let bits = v.values();
                let bm: Bitmap = (0..n)
                    .map(|i| {
                        f(i64::from(
                            bits.get_bit(i),
                        )) != 0
                    })
                    .collect();
                ColumnData::Bool(
                    BooleanArray::new(
                        ArrowDataType::Boolean,
                        bm,
                        None,
                    ),
                )
            }
            ColumnData::F32(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as f32
                })
                .map(ColumnData::F32)
                .unwrap()
            }
            ColumnData::F64(v) => {
                primitive_map(v, |x| {
                    f(x as i64) as f64
                })
                .map(ColumnData::F64)
                .unwrap()
            }
            ColumnData::Str(_)
            | ColumnData::Bin(_) => panic!(
                "map_i64 is not supported for string/binary ColumnData"
            ),
        }
    }

    /// Gather elements by computing indices on-the-fly from a closure.
    /// Avoids allocating a separate index vector before gathering.
    pub(crate) fn gather_by(
        &self,
        len: usize,
        index_fn: impl Fn(usize) -> usize,
    ) -> ColumnData {
        match self {
            ColumnData::Bool(v) => {
                let bits = v.values();
                let bm: Bitmap = (0..len)
                    .map(|i| {
                        bits.get_bit(index_fn(i))
                    })
                    .collect();
                ColumnData::Bool(
                    BooleanArray::new(
                        ArrowDataType::Boolean,
                        bm,
                        None,
                    ),
                )
            }
            ColumnData::I8(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::I8)
                .unwrap()
            }
            ColumnData::I16(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::I16)
                .unwrap()
            }
            ColumnData::I32(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::I32)
                .unwrap()
            }
            ColumnData::I64(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::I64)
                .unwrap()
            }
            ColumnData::U8(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::U8)
                .unwrap()
            }
            ColumnData::U16(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::U16)
                .unwrap()
            }
            ColumnData::U32(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::U32)
                .unwrap()
            }
            ColumnData::U64(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::U64)
                .unwrap()
            }
            ColumnData::F32(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::F32)
                .unwrap()
            }
            ColumnData::F64(v) => {
                primitive_gather(
                    v,
                    (0..len).map(&index_fn),
                )
                .map(ColumnData::F64)
                .unwrap()
            }
            ColumnData::Str(v) => {
                ColumnData::Str(gather_utf8(
                    v,
                    (0..len).map(&index_fn),
                ))
            }
            ColumnData::Bin(v) => {
                ColumnData::Bin(gather_binary(
                    v,
                    (0..len).map(&index_fn),
                ))
            }
        }
    }

    /// Produce a column by repeating each element `inner_repeat` times,
    /// then tiling the pattern `tile_count` times.
    pub(crate) fn repeat_tile(
        &self,
        inner_repeat: usize,
        tile_count: usize,
    ) -> ColumnData {
        match self {
            ColumnData::Bool(v) => {
                let bits = v.values();
                let n = v.len();
                let total =
                    n * inner_repeat * tile_count;
                let mut bm =
                    MutableBitmap::with_capacity(
                        total,
                    );
                for _ in 0..tile_count {
                    for i in 0..n {
                        let bit = bits.get_bit(i);
                        for _ in 0..inner_repeat {
                            bm.push(bit);
                        }
                    }
                }
                ColumnData::Bool(
                    BooleanArray::new(
                        ArrowDataType::Boolean,
                        bm.into(),
                        None,
                    ),
                )
            }
            ColumnData::I8(v) => ColumnData::I8(
                primitive_from_vec(
                    repeat_tile_slice(
                        v.values(),
                        inner_repeat,
                        tile_count,
                    ),
                ),
            ),
            ColumnData::I16(v) => {
                ColumnData::I16(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::I64(v) => {
                ColumnData::I64(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::U8(v) => ColumnData::U8(
                primitive_from_vec(
                    repeat_tile_slice(
                        v.values(),
                        inner_repeat,
                        tile_count,
                    ),
                ),
            ),
            ColumnData::U16(v) => {
                ColumnData::U16(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::U64(v) => {
                ColumnData::U64(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    primitive_from_vec(
                        repeat_tile_slice(
                            v.values(),
                            inner_repeat,
                            tile_count,
                        ),
                    ),
                )
            }
            ColumnData::Str(v) => {
                ColumnData::Str(repeat_tile_utf8(
                    v,
                    inner_repeat,
                    tile_count,
                ))
            }
            ColumnData::Bin(v) => {
                ColumnData::Bin(
                    repeat_tile_binary(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
        }
    }

    pub(crate) fn into_series(
        self,
        name: &str,
    ) -> Series {
        let array: Box<dyn Array> = match self {
            ColumnData::Bool(v) => Box::new(v),
            ColumnData::I8(v) => Box::new(v),
            ColumnData::I16(v) => Box::new(v),
            ColumnData::I32(v) => Box::new(v),
            ColumnData::I64(v) => Box::new(v),
            ColumnData::U8(v) => Box::new(v),
            ColumnData::U16(v) => Box::new(v),
            ColumnData::U32(v) => Box::new(v),
            ColumnData::U64(v) => Box::new(v),
            ColumnData::F32(v) => Box::new(v),
            ColumnData::F64(v) => Box::new(v),
            ColumnData::Str(v) => Box::new(v),
            ColumnData::Bin(v) => Box::new(v),
        };
        Series::from_arrow(name.into(), array)
            .expect("ColumnData arrow array always builds a valid Series")
    }

    pub(crate) fn borrow_into_series(
        &self,
        name: &str,
    ) -> Series {
        // Cloning Arrow arrays is `O(1)` (refcount bump), so this is
        // equivalent in cost to the old `borrow_into_series` while
        // avoiding the buffer clone the old `Series::new` triggered.
        self.clone().into_series(name)
    }

    /// Decode CF scale/offset packing: `raw * scale + offset`.
    /// Elements matching `fill_value` (in raw space) become NaN.
    pub(crate) fn to_f64_scaled(
        &self,
        scale: f64,
        offset: f64,
        fill_value: Option<f64>,
    ) -> ColumnData {
        macro_rules! decode_primitive {
            ($v:expr) => {{
                let arr: &PrimitiveArray<_> = $v;
                let out: Vec<f64> = arr
                    .values()
                    .iter()
                    .map(|&x| {
                        let raw = x as f64;
                        if fill_value.is_some_and(
                            |fv| raw == fv,
                        ) {
                            f64::NAN
                        } else {
                            raw * scale + offset
                        }
                    })
                    .collect();
                ColumnData::F64(
                    primitive_from_vec(out),
                )
            }};
        }
        match self {
            ColumnData::Bool(v) => {
                let bits = v.values();
                let n = v.len();
                let out: Vec<f64> = (0..n)
                    .map(|i| {
                        let raw = u8::from(
                            bits.get_bit(i),
                        )
                            as f64;
                        if fill_value.is_some_and(
                            |fv| raw == fv,
                        ) {
                            f64::NAN
                        } else {
                            raw * scale + offset
                        }
                    })
                    .collect();
                ColumnData::F64(
                    primitive_from_vec(out),
                )
            }
            ColumnData::I8(v) => {
                decode_primitive!(v)
            }
            ColumnData::I16(v) => {
                decode_primitive!(v)
            }
            ColumnData::I32(v) => {
                decode_primitive!(v)
            }
            ColumnData::I64(v) => {
                decode_primitive!(v)
            }
            ColumnData::U8(v) => {
                decode_primitive!(v)
            }
            ColumnData::U16(v) => {
                decode_primitive!(v)
            }
            ColumnData::U32(v) => {
                decode_primitive!(v)
            }
            ColumnData::U64(v) => {
                decode_primitive!(v)
            }
            ColumnData::F32(v) => {
                decode_primitive!(v)
            }
            ColumnData::F64(v) => {
                decode_primitive!(v)
            }
            ColumnData::Str(_)
            | ColumnData::Bin(_) => panic!(
                "to_f64_scaled is not supported for string/binary ColumnData"
            ),
        }
    }

    /// Concatenate this `ColumnData` with another, returning a new value.
    /// Panics if types don't match.
    pub(crate) fn concat(
        self,
        other: &ColumnData,
    ) -> ColumnData {
        match (self, other) {
            (
                ColumnData::Bool(a),
                ColumnData::Bool(b),
            ) => {
                let mut bm =
                    MutableBitmap::with_capacity(
                        a.len() + b.len(),
                    );
                let av = a.values();
                for i in 0..a.len() {
                    bm.push(av.get_bit(i));
                }
                let bv = b.values();
                for i in 0..b.len() {
                    bm.push(bv.get_bit(i));
                }
                ColumnData::Bool(
                    BooleanArray::new(
                        ArrowDataType::Boolean,
                        bm.into(),
                        None,
                    ),
                )
            }
            (
                ColumnData::I8(a),
                ColumnData::I8(b),
            ) => ColumnData::I8(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::I16(a),
                ColumnData::I16(b),
            ) => ColumnData::I16(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::I32(a),
                ColumnData::I32(b),
            ) => ColumnData::I32(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::I64(a),
                ColumnData::I64(b),
            ) => ColumnData::I64(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::U8(a),
                ColumnData::U8(b),
            ) => ColumnData::U8(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::U16(a),
                ColumnData::U16(b),
            ) => ColumnData::U16(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::U32(a),
                ColumnData::U32(b),
            ) => ColumnData::U32(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::U64(a),
                ColumnData::U64(b),
            ) => ColumnData::U64(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::F32(a),
                ColumnData::F32(b),
            ) => ColumnData::F32(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::F64(a),
                ColumnData::F64(b),
            ) => ColumnData::F64(
                primitive_concat(&a, b),
            ),
            (
                ColumnData::Str(a),
                ColumnData::Str(b),
            ) => ColumnData::Str(concat_utf8(
                &a, b,
            )),
            (
                ColumnData::Bin(a),
                ColumnData::Bin(b),
            ) => ColumnData::Bin(concat_binary(
                &a, b,
            )),
            _ => panic!(
                "ColumnData::concat type mismatch"
            ),
        }
    }
}

// ============================================================================
// PrimitiveArray helpers
// ============================================================================

#[inline]
fn primitive_from_vec<T>(
    vs: Vec<T>,
) -> PrimitiveArray<T>
where
    T: polars_arrow::types::NativeType,
{
    PrimitiveArray::from_vec(vs)
}

/// Gather a primitive array by an index iterator. The index iterator may
/// be a single-pass iterator; the result is collected into a new `Vec`.
fn primitive_gather<T>(
    src: &PrimitiveArray<T>,
    indices: impl Iterator<Item = usize>,
) -> Result<PrimitiveArray<T>, &'static str>
where
    T: polars_arrow::types::NativeType,
{
    let values = src.values();
    let v: Vec<T> =
        indices.map(|i| values[i]).collect();
    Ok(primitive_from_vec(v))
}

/// Map a primitive array element-wise. The output type may differ.
fn primitive_map<T, U, F>(
    src: &PrimitiveArray<T>,
    f: F,
) -> Result<PrimitiveArray<U>, &'static str>
where
    T: polars_arrow::types::NativeType,
    U: polars_arrow::types::NativeType,
    F: Fn(T) -> U,
{
    let v: Vec<U> = src
        .values()
        .iter()
        .map(|&x| f(x))
        .collect();
    Ok(primitive_from_vec(v))
}

fn primitive_concat<T>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> PrimitiveArray<T>
where
    T: polars_arrow::types::NativeType,
{
    let mut out: Vec<T> =
        Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a.values());
    out.extend_from_slice(b.values());
    primitive_from_vec(out)
}

fn bitmap_from_indexed(
    src: &Bitmap,
    indices: impl Iterator<Item = usize>,
) -> Bitmap {
    let (lower, upper) = indices.size_hint();
    let cap = upper.unwrap_or(lower);
    let mut bm =
        MutableBitmap::with_capacity(cap);
    for i in indices {
        bm.push(src.get_bit(i));
    }
    bm.into()
}

// ============================================================================
// Variable-length helpers (Utf8Array / BinaryArray, large i64 offsets)
// ============================================================================

/// Build a `Utf8Array<i64>` from a `(values, offsets)` pair. The
/// `try_new` path validates UTF-8; we only call this when the source
/// already guarantees it (the input was a `Utf8Array<i64>` we are
/// re-shuffling, or a freshly decoded `string` chunk validated by
/// zarrs's `String::from_array_bytes`).
fn make_utf8(
    values: Vec<u8>,
    offsets_i64: Vec<i64>,
) -> Utf8Array<i64> {
    let offsets: OffsetsBuffer<i64> =
        OffsetsBuffer::<i64>::try_from(
            offsets_i64,
        )
        .expect("monotonic i64 offsets");
    // SAFETY: input bytes came from a validated `Utf8Array<i64>` /
    // `String::from_array_bytes`, so each `[off[i]..off[i+1]]` slice is
    // valid UTF-8.
    unsafe {
        Utf8Array::<i64>::new_unchecked(
            ArrowDataType::LargeUtf8,
            offsets,
            values.into(),
            None,
        )
    }
}

fn make_binary(
    values: Vec<u8>,
    offsets_i64: Vec<i64>,
) -> BinaryArray<i64> {
    let offsets: OffsetsBuffer<i64> =
        OffsetsBuffer::<i64>::try_from(
            offsets_i64,
        )
        .expect("monotonic i64 offsets");
    BinaryArray::<i64>::new(
        ArrowDataType::LargeBinary,
        offsets,
        values.into(),
        None,
    )
}

fn gather_utf8(
    src: &Utf8Array<i64>,
    indices: impl Iterator<Item = usize>,
) -> Utf8Array<i64> {
    let src_values = src.values();
    let src_offsets = src.offsets();
    let (lower, upper) = indices.size_hint();
    let cap = upper.unwrap_or(lower);
    let mut new_values: Vec<u8> = Vec::new();
    let mut new_offsets: Vec<i64> =
        Vec::with_capacity(cap + 1);
    new_offsets.push(0);
    for i in indices {
        let s = src_offsets[i] as usize;
        let e = src_offsets[i + 1] as usize;
        new_values
            .extend_from_slice(&src_values[s..e]);
        new_offsets.push(new_values.len() as i64);
    }
    make_utf8(new_values, new_offsets)
}

fn gather_binary(
    src: &BinaryArray<i64>,
    indices: impl Iterator<Item = usize>,
) -> BinaryArray<i64> {
    let src_values = src.values();
    let src_offsets = src.offsets();
    let (lower, upper) = indices.size_hint();
    let cap = upper.unwrap_or(lower);
    let mut new_values: Vec<u8> = Vec::new();
    let mut new_offsets: Vec<i64> =
        Vec::with_capacity(cap + 1);
    new_offsets.push(0);
    for i in indices {
        let s = src_offsets[i] as usize;
        let e = src_offsets[i + 1] as usize;
        new_values
            .extend_from_slice(&src_values[s..e]);
        new_offsets.push(new_values.len() as i64);
    }
    make_binary(new_values, new_offsets)
}

fn repeat_tile_utf8(
    src: &Utf8Array<i64>,
    inner_repeat: usize,
    tile_count: usize,
) -> Utf8Array<i64> {
    if src.is_empty()
        || inner_repeat == 0
        || tile_count == 0
    {
        return Utf8Array::<i64>::default();
    }
    let n = src.len();
    let total = n * inner_repeat * tile_count;
    let src_values = src.values();
    let src_offsets = src.offsets();
    let mut new_values: Vec<u8> = Vec::new();
    let mut new_offsets: Vec<i64> =
        Vec::with_capacity(total + 1);
    new_offsets.push(0);
    for _ in 0..tile_count {
        for i in 0..n {
            let s = src_offsets[i] as usize;
            let e = src_offsets[i + 1] as usize;
            for _ in 0..inner_repeat {
                new_values.extend_from_slice(
                    &src_values[s..e],
                );
                new_offsets.push(new_values.len() as i64);
            }
        }
    }
    make_utf8(new_values, new_offsets)
}

fn repeat_tile_binary(
    src: &BinaryArray<i64>,
    inner_repeat: usize,
    tile_count: usize,
) -> BinaryArray<i64> {
    if src.is_empty()
        || inner_repeat == 0
        || tile_count == 0
    {
        return BinaryArray::<i64>::new_empty(
            ArrowDataType::LargeBinary,
        );
    }
    let n = src.len();
    let total = n * inner_repeat * tile_count;
    let src_values = src.values();
    let src_offsets = src.offsets();
    let mut new_values: Vec<u8> = Vec::new();
    let mut new_offsets: Vec<i64> =
        Vec::with_capacity(total + 1);
    new_offsets.push(0);
    for _ in 0..tile_count {
        for i in 0..n {
            let s = src_offsets[i] as usize;
            let e = src_offsets[i + 1] as usize;
            for _ in 0..inner_repeat {
                new_values.extend_from_slice(
                    &src_values[s..e],
                );
                new_offsets.push(new_values.len() as i64);
            }
        }
    }
    make_binary(new_values, new_offsets)
}

fn concat_utf8(
    a: &Utf8Array<i64>,
    b: &Utf8Array<i64>,
) -> Utf8Array<i64> {
    let av = a.values().as_slice();
    let ao = a.offsets();
    let bv = b.values().as_slice();
    let bo = b.offsets();
    let total_n = a.len() + b.len();
    let total_bytes = av.len() + bv.len();
    let mut new_values: Vec<u8> =
        Vec::with_capacity(total_bytes);
    let mut new_offsets: Vec<i64> =
        Vec::with_capacity(total_n + 1);
    new_offsets.push(0);
    for i in 0..a.len() {
        let s = ao[i] as usize;
        let e = ao[i + 1] as usize;
        new_values.extend_from_slice(&av[s..e]);
        new_offsets.push(new_values.len() as i64);
    }
    for i in 0..b.len() {
        let s = bo[i] as usize;
        let e = bo[i + 1] as usize;
        new_values.extend_from_slice(&bv[s..e]);
        new_offsets.push(new_values.len() as i64);
    }
    make_utf8(new_values, new_offsets)
}

fn concat_binary(
    a: &BinaryArray<i64>,
    b: &BinaryArray<i64>,
) -> BinaryArray<i64> {
    let av = a.values().as_slice();
    let ao = a.offsets();
    let bv = b.values().as_slice();
    let bo = b.offsets();
    let total_n = a.len() + b.len();
    let total_bytes = av.len() + bv.len();
    let mut new_values: Vec<u8> =
        Vec::with_capacity(total_bytes);
    let mut new_offsets: Vec<i64> =
        Vec::with_capacity(total_n + 1);
    new_offsets.push(0);
    for i in 0..a.len() {
        let s = ao[i] as usize;
        let e = ao[i + 1] as usize;
        new_values.extend_from_slice(&av[s..e]);
        new_offsets.push(new_values.len() as i64);
    }
    for i in 0..b.len() {
        let s = bo[i] as usize;
        let e = bo[i + 1] as usize;
        new_values.extend_from_slice(&bv[s..e]);
        new_offsets.push(new_values.len() as i64);
    }
    make_binary(new_values, new_offsets)
}

// ============================================================================
// FromArrayBytes wrapper — implements zarrs's decode-into-T trait
// ============================================================================

/// Wrapper that implements [`FromArrayBytes`] for [`ColumnData`] by
/// dispatching on the zarr `DataType` name. This is the entry point used
/// by the sync/async chunk retrieval functions, replacing per-dtype
/// `Vec<T>` decodes with a single typed Arrow decode.
pub(crate) struct DecodedChunk(pub ColumnData);

impl FromArrayBytes for DecodedChunk {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let name = data_type
            .name(ZarrVersion::V3)
            .map(|s| s.into_owned())
            .unwrap_or_else(|| {
                "binary".to_string()
            });
        let cd = decode_to_column(
            name.as_str(),
            bytes,
        )?;
        Ok(DecodedChunk(cd))
    }

    fn from_array_bytes_arc(
        bytes: Arc<ArrayBytes<'static>>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        Self::from_array_bytes(
            Arc::unwrap_or_clone(bytes),
            shape,
            data_type,
        )
    }
}

fn decode_to_column(
    dtype_name: &str,
    bytes: ArrayBytes<'static>,
) -> Result<ColumnData, ArrayError> {
    match dtype_name {
        "bool" => Ok(ColumnData::Bool(
            decode_bool(bytes)?,
        )),
        "int8" => Ok(ColumnData::I8(
            decode_primitive::<i8>(bytes)?,
        )),
        "int16" => Ok(ColumnData::I16(
            decode_primitive::<i16>(bytes)?,
        )),
        "int32" => Ok(ColumnData::I32(
            decode_primitive::<i32>(bytes)?,
        )),
        "int64" => Ok(ColumnData::I64(
            decode_primitive::<i64>(bytes)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            decode_primitive::<u8>(bytes)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            decode_primitive::<u16>(bytes)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            decode_primitive::<u32>(bytes)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            decode_primitive::<u64>(bytes)?,
        )),
        "float32" => Ok(ColumnData::F32(
            decode_primitive::<f32>(bytes)?,
        )),
        "float64" => Ok(ColumnData::F64(
            decode_primitive::<f64>(bytes)?,
        )),
        "string" => Ok(ColumnData::Str(
            decode_utf8(bytes)?,
        )),
        "bytes" => Ok(ColumnData::Bin(
            decode_binary(bytes)?,
        )),
        other => Err(ArrayError::Other(format!(
            "unsupported zarr dtype: {other}"
        ))),
    }
}

fn decode_primitive<T>(
    bytes: ArrayBytes<'static>,
) -> Result<PrimitiveArray<T>, ArrayError>
where
    T: bytemuck::Pod
        + polars_arrow::types::NativeType,
{
    let raw =
        bytes.into_fixed().map_err(|e| {
            ArrayError::Other(e.to_string())
        })?;
    let bytes_vec: Vec<u8> = raw.into_owned();
    let values: Vec<T> =
        transmute_from_bytes_vec::<T>(bytes_vec);
    Ok(PrimitiveArray::from_vec(values))
}

fn decode_bool(
    bytes: ArrayBytes<'static>,
) -> Result<BooleanArray, ArrayError> {
    let raw =
        bytes.into_fixed().map_err(|e| {
            ArrayError::Other(e.to_string())
        })?;
    // zarrs encodes bool as one byte per element (`0`/`1`); validate the
    // values by the same rule `bool::from_array_bytes` uses, then
    // bit-pack into an Arrow bitmap.
    let bytes_u8: Vec<u8> = match raw {
        std::borrow::Cow::Borrowed(b) => {
            convert_from_bytes_slice::<u8>(b)
        }
        std::borrow::Cow::Owned(v) => v,
    };
    if !bytes_u8.iter().all(|&u| u <= 1) {
        return Err(ArrayError::Other(
            "invalid bool element value"
                .to_string(),
        ));
    }
    let mut bm = MutableBitmap::with_capacity(
        bytes_u8.len(),
    );
    for b in &bytes_u8 {
        bm.push(*b != 0);
    }
    Ok(BooleanArray::new(
        ArrowDataType::Boolean,
        bm.into(),
        None,
    ))
}

fn decode_utf8(
    bytes: ArrayBytes<'static>,
) -> Result<Utf8Array<i64>, ArrayError> {
    let var =
        bytes.into_variable().map_err(|e| {
            ArrayError::Other(e.to_string())
        })?;
    let (raw, offsets) = var.into_parts();
    let bytes_vec: Vec<u8> = raw.into_owned();
    let offsets_i64: Vec<i64> = offsets
        .iter()
        .map(|&o| o as i64)
        .collect();
    let offsets_buf: OffsetsBuffer<i64> =
        OffsetsBuffer::<i64>::try_from(
            offsets_i64,
        )
        .map_err(|e| {
            ArrayError::Other(e.to_string())
        })?;
    Utf8Array::<i64>::try_new(
        ArrowDataType::LargeUtf8,
        offsets_buf,
        bytes_vec.into(),
        None,
    )
    .map_err(|e| ArrayError::Other(e.to_string()))
}

fn decode_binary(
    bytes: ArrayBytes<'static>,
) -> Result<BinaryArray<i64>, ArrayError> {
    let var =
        bytes.into_variable().map_err(|e| {
            ArrayError::Other(e.to_string())
        })?;
    let (raw, offsets) = var.into_parts();
    let bytes_vec: Vec<u8> = raw.into_owned();
    let offsets_i64: Vec<i64> = offsets
        .iter()
        .map(|&o| o as i64)
        .collect();
    let offsets_buf: OffsetsBuffer<i64> =
        OffsetsBuffer::<i64>::try_from(
            offsets_i64,
        )
        .map_err(|e| {
            ArrayError::Other(e.to_string())
        })?;
    BinaryArray::<i64>::try_new(
        ArrowDataType::LargeBinary,
        offsets_buf,
        bytes_vec.into(),
        None,
    )
    .map_err(|e| ArrayError::Other(e.to_string()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_i64_vec_round_trip() {
        let cd = ColumnData::from_i64_vec(vec![
            1, 2, 3, 4,
        ]);
        assert_eq!(cd.len(), 4);
        assert_eq!(
            cd.as_i64_values().unwrap(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn slice_zero_copy_semantics() {
        let cd = ColumnData::from_i64_vec(
            (0..10i64).collect(),
        );
        let s = cd.slice(2, 5);
        assert_eq!(
            s.as_i64_values().unwrap(),
            &[2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn take_indices_primitive() {
        let cd = ColumnData::from_i64_vec(vec![
            10, 20, 30, 40,
        ]);
        let t = cd.take_indices(&[3, 0, 2]);
        assert_eq!(
            t.as_i64_values().unwrap(),
            &[40, 10, 30]
        );
    }

    #[test]
    fn gather_by_primitive() {
        let cd = ColumnData::from_i64_vec(vec![
            5, 6, 7, 8,
        ]);
        let g = cd.gather_by(6, |i| i % 4);
        assert_eq!(
            g.as_i64_values().unwrap(),
            &[5, 6, 7, 8, 5, 6]
        );
    }

    #[test]
    fn repeat_tile_primitive() {
        let cd =
            ColumnData::from_i64_vec(vec![1, 2]);
        let r = cd.repeat_tile(3, 2);
        assert_eq!(
            r.as_i64_values().unwrap(),
            &[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
        );
    }

    #[test]
    fn concat_primitive() {
        let a =
            ColumnData::from_i64_vec(vec![1, 2]);
        let b = ColumnData::from_i64_vec(vec![
            3, 4, 5,
        ]);
        let c = a.concat(&b);
        assert_eq!(
            c.as_i64_values().unwrap(),
            &[1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn map_i64_round_trip() {
        let cd = ColumnData::from_i64_vec(vec![
            1, 2, 3,
        ]);
        let m = cd.map_i64(|x| x * 10);
        assert_eq!(
            m.as_i64_values().unwrap(),
            &[10, 20, 30]
        );
    }

    #[test]
    fn to_f64_scaled_basic() {
        let cd = ColumnData::I32(
            PrimitiveArray::from_vec(vec![
                1i32, 2, -1,
            ]),
        );
        let f = cd.to_f64_scaled(
            0.5,
            10.0,
            Some(-1.0),
        );
        let vs = f.as_f64_values().unwrap();
        assert!((vs[0] - 10.5).abs() < 1e-9);
        assert!((vs[1] - 11.0).abs() < 1e-9);
        assert!(vs[2].is_nan());
    }

    #[test]
    fn bool_repeat_tile_and_gather() {
        let bm: Bitmap = [true, false, true]
            .into_iter()
            .collect();
        let cd =
            ColumnData::Bool(BooleanArray::new(
                ArrowDataType::Boolean,
                bm,
                None,
            ));
        let r = cd.repeat_tile(2, 1);
        // expect: T T F F T T
        let ColumnData::Bool(arr) = &r else {
            panic!("wrong variant");
        };
        let bits = arr.values();
        let got: Vec<bool> = (0..arr.len())
            .map(|i| bits.get_bit(i))
            .collect();
        assert_eq!(
            got,
            vec![
                true, true, false, false, true,
                true
            ]
        );

        let g = cd.gather_by(2, |i| i);
        let ColumnData::Bool(arr) = &g else {
            panic!("wrong variant");
        };
        let bits = arr.values();
        let got: Vec<bool> = (0..arr.len())
            .map(|i| bits.get_bit(i))
            .collect();
        assert_eq!(got, vec![true, false]);
    }

    #[test]
    fn utf8_take_and_repeat() {
        let arr = Utf8Array::<i64>::from([
            Some("alpha"),
            Some("beta"),
            Some("gamma"),
        ]);
        let cd = ColumnData::Str(arr);
        let t = cd.take_indices(&[2, 0]);
        let ColumnData::Str(out) = &t else {
            panic!("wrong variant");
        };
        assert_eq!(out.value(0), "gamma");
        assert_eq!(out.value(1), "alpha");

        let r = cd.repeat_tile(2, 1);
        let ColumnData::Str(out) = &r else {
            panic!("wrong variant");
        };
        assert_eq!(out.len(), 6);
        assert_eq!(out.value(0), "alpha");
        assert_eq!(out.value(1), "alpha");
        assert_eq!(out.value(2), "beta");
        assert_eq!(out.value(3), "beta");
        assert_eq!(out.value(4), "gamma");
        assert_eq!(out.value(5), "gamma");
    }
}
