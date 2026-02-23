use polars::prelude::{NamedFrom, Series};

/// Build a Vec by repeating each element of `src` `inner_repeat`
/// times, then tiling the resulting pattern `tile_count` times.
///
/// Total output length = `src.len() * inner_repeat * tile_count`.
///
/// Semantically equivalent to:
/// ```ignore
/// (0..total).map(|i| src[(i / inner_repeat) % src.len()]).collect()
/// ```
/// but uses only O(log n) memcpy operations instead of per-element
/// integer division, giving a large speedup on the hot path.
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
        // No inner repeat — just copy source
        // values as the first tile.
        output.extend_from_slice(src);
    } else {
        // Build one tile: for each value, fill
        // via doubling memcpy (O(log inner_repeat)
        // copies per value instead of inner_repeat
        // scalar writes).
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

    // Tile to full output via memcpy
    if tile_count > 1 {
        for _ in 1..tile_count {
            output
                .extend_from_within(0..tile_len);
        }
    }

    output
}

#[derive(Debug)]
pub(crate) enum ColumnData {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

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
        }
    }

    pub(crate) fn slice(
        &self,
        start: usize,
        len: usize,
    ) -> ColumnData {
        let end = start + len;
        match self {
            ColumnData::Bool(v) => {
                ColumnData::Bool(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::I8(v) => ColumnData::I8(
                v[start..end].to_vec(),
            ),
            ColumnData::I16(v) => {
                ColumnData::I16(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::I64(v) => {
                ColumnData::I64(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::U8(v) => ColumnData::U8(
                v[start..end].to_vec(),
            ),
            ColumnData::U16(v) => {
                ColumnData::U16(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::U64(v) => {
                ColumnData::U64(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    v[start..end].to_vec(),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    v[start..end].to_vec(),
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
                ColumnData::Bool(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::I8(v) => ColumnData::I8(
                indices
                    .iter()
                    .map(|&i| v[i])
                    .collect(),
            ),
            ColumnData::I16(v) => {
                ColumnData::I16(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::I64(v) => {
                ColumnData::I64(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::U8(v) => ColumnData::U8(
                indices
                    .iter()
                    .map(|&i| v[i])
                    .collect(),
            ),
            ColumnData::U16(v) => {
                ColumnData::U16(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::U64(v) => {
                ColumnData::U64(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    indices
                        .iter()
                        .map(|&i| v[i])
                        .collect(),
                )
            }
        }
    }

    pub(crate) fn get_i64(
        &self,
        idx: usize,
    ) -> Option<i64> {
        match self {
            ColumnData::I64(v) => Some(v[idx]),
            ColumnData::I32(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::I16(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::I8(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::U64(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::U32(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::U16(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::U8(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::F32(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::F64(v) => {
                Some(v[idx] as i64)
            }
            ColumnData::Bool(v) => {
                Some(i64::from(v[idx]))
            }
        }
    }

    pub(crate) fn map_i64(
        &self,
        f: impl Fn(i64) -> i64,
    ) -> ColumnData {
        match self {
            ColumnData::I64(v) => {
                ColumnData::I64(
                    v.iter()
                        .map(|&x| f(x))
                        .collect(),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as i32
                        })
                        .collect(),
                )
            }
            ColumnData::I16(v) => {
                ColumnData::I16(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as i16
                        })
                        .collect(),
                )
            }
            ColumnData::I8(v) => ColumnData::I8(
                v.iter()
                    .map(|&x| f(x as i64) as i8)
                    .collect(),
            ),
            ColumnData::U64(v) => {
                ColumnData::U64(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as u64
                        })
                        .collect(),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as u32
                        })
                        .collect(),
                )
            }
            ColumnData::U16(v) => {
                ColumnData::U16(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as u16
                        })
                        .collect(),
                )
            }
            ColumnData::U8(v) => ColumnData::U8(
                v.iter()
                    .map(|&x| f(x as i64) as u8)
                    .collect(),
            ),
            ColumnData::Bool(v) => {
                ColumnData::Bool(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) != 0
                        })
                        .collect(),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as f32
                        })
                        .collect(),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    v.iter()
                        .map(|&x| {
                            f(x as i64) as f64
                        })
                        .collect(),
                )
            }
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
                ColumnData::Bool(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::I8(v) => ColumnData::I8(
                (0..len)
                    .map(|i| v[index_fn(i)])
                    .collect(),
            ),
            ColumnData::I16(v) => {
                ColumnData::I16(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::I64(v) => {
                ColumnData::I64(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::U8(v) => ColumnData::U8(
                (0..len)
                    .map(|i| v[index_fn(i)])
                    .collect(),
            ),
            ColumnData::U16(v) => {
                ColumnData::U16(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::U64(v) => {
                ColumnData::U64(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    (0..len)
                        .map(|i| v[index_fn(i)])
                        .collect(),
                )
            }
        }
    }

    /// Produce a column by repeating each element
    /// `inner_repeat` times, then tiling the pattern
    /// `tile_count` times. Uses only memcpy operations.
    ///
    /// Equivalent to:
    /// ```ignore
    /// self.gather_by(
    ///   self.len() * inner_repeat * tile_count,
    ///   |i| (i / inner_repeat) % self.len(),
    /// )
    /// ```
    /// but avoids all per-element integer division.
    pub(crate) fn repeat_tile(
        &self,
        inner_repeat: usize,
        tile_count: usize,
    ) -> ColumnData {
        match self {
            ColumnData::Bool(v) => {
                ColumnData::Bool(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::I8(v) => {
                ColumnData::I8(repeat_tile_slice(
                    v,
                    inner_repeat,
                    tile_count,
                ))
            }
            ColumnData::I16(v) => {
                ColumnData::I16(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::I32(v) => {
                ColumnData::I32(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::I64(v) => {
                ColumnData::I64(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::U8(v) => {
                ColumnData::U8(repeat_tile_slice(
                    v,
                    inner_repeat,
                    tile_count,
                ))
            }
            ColumnData::U16(v) => {
                ColumnData::U16(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::U32(v) => {
                ColumnData::U32(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::U64(v) => {
                ColumnData::U64(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::F32(v) => {
                ColumnData::F32(
                    repeat_tile_slice(
                        v,
                        inner_repeat,
                        tile_count,
                    ),
                )
            }
            ColumnData::F64(v) => {
                ColumnData::F64(
                    repeat_tile_slice(
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
        match self {
            ColumnData::Bool(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I8(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I16(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I32(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I64(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U8(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U16(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U32(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U64(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::F32(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::F64(v) => {
                Series::new(name.into(), v)
            }
        }
    }

    pub(crate) fn borrow_into_series(
        &self,
        name: &str,
    ) -> Series {
        match self {
            ColumnData::Bool(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I8(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I16(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I32(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::I64(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U8(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U16(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U32(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::U64(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::F32(v) => {
                Series::new(name.into(), v)
            }
            ColumnData::F64(v) => {
                Series::new(name.into(), v)
            }
        }
    }

    /// Concatenate this ColumnData with another, returning a new value.
    /// Panics if types don't match.
    pub(crate) fn concat(
        self,
        other: &ColumnData,
    ) -> ColumnData {
        match (self, other) {
            (
                ColumnData::Bool(mut a),
                ColumnData::Bool(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::Bool(a)
            }
            (
                ColumnData::I8(mut a),
                ColumnData::I8(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::I8(a)
            }
            (
                ColumnData::I16(mut a),
                ColumnData::I16(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::I16(a)
            }
            (
                ColumnData::I32(mut a),
                ColumnData::I32(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::I32(a)
            }
            (
                ColumnData::I64(mut a),
                ColumnData::I64(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::I64(a)
            }
            (
                ColumnData::U8(mut a),
                ColumnData::U8(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::U8(a)
            }
            (
                ColumnData::U16(mut a),
                ColumnData::U16(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::U16(a)
            }
            (
                ColumnData::U32(mut a),
                ColumnData::U32(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::U32(a)
            }
            (
                ColumnData::U64(mut a),
                ColumnData::U64(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::U64(a)
            }
            (
                ColumnData::F32(mut a),
                ColumnData::F32(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::F32(a)
            }
            (
                ColumnData::F64(mut a),
                ColumnData::F64(b),
            ) => {
                a.extend_from_slice(b);
                ColumnData::F64(a)
            }
            _ => panic!(
                "ColumnData::concat type mismatch"
            ),
        }
    }
}
