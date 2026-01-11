use polars::prelude::{NamedFrom, Series};

#[derive(Debug, Clone)]
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

    pub(crate) fn slice(&self, start: usize, len: usize) -> ColumnData {
        let end = start + len;
        match self {
            ColumnData::Bool(v) => ColumnData::Bool(v[start..end].to_vec()),
            ColumnData::I8(v) => ColumnData::I8(v[start..end].to_vec()),
            ColumnData::I16(v) => ColumnData::I16(v[start..end].to_vec()),
            ColumnData::I32(v) => ColumnData::I32(v[start..end].to_vec()),
            ColumnData::I64(v) => ColumnData::I64(v[start..end].to_vec()),
            ColumnData::U8(v) => ColumnData::U8(v[start..end].to_vec()),
            ColumnData::U16(v) => ColumnData::U16(v[start..end].to_vec()),
            ColumnData::U32(v) => ColumnData::U32(v[start..end].to_vec()),
            ColumnData::U64(v) => ColumnData::U64(v[start..end].to_vec()),
            ColumnData::F32(v) => ColumnData::F32(v[start..end].to_vec()),
            ColumnData::F64(v) => ColumnData::F64(v[start..end].to_vec()),
        }
    }

    pub(crate) fn take_indices(&self, indices: &[usize]) -> ColumnData {
        match self {
            ColumnData::Bool(v) => ColumnData::Bool(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I8(v) => ColumnData::I8(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I16(v) => ColumnData::I16(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I32(v) => ColumnData::I32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I64(v) => ColumnData::I64(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U8(v) => ColumnData::U8(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U16(v) => ColumnData::U16(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U32(v) => ColumnData::U32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U64(v) => ColumnData::U64(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::F32(v) => ColumnData::F32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::F64(v) => ColumnData::F64(indices.iter().map(|&i| v[i]).collect()),
        }
    }

    pub(crate) fn get_f64(&self, idx: usize) -> Option<f64> {
        match self {
            ColumnData::F64(v) => Some(v[idx]),
            ColumnData::F32(v) => Some(v[idx] as f64),
            ColumnData::I64(v) => Some(v[idx] as f64),
            ColumnData::I32(v) => Some(v[idx] as f64),
            ColumnData::I16(v) => Some(v[idx] as f64),
            ColumnData::I8(v) => Some(v[idx] as f64),
            ColumnData::U64(v) => Some(v[idx] as f64),
            ColumnData::U32(v) => Some(v[idx] as f64),
            ColumnData::U16(v) => Some(v[idx] as f64),
            ColumnData::U8(v) => Some(v[idx] as f64),
            ColumnData::Bool(_) => None,
        }
    }

    pub(crate) fn get_i64(&self, idx: usize) -> Option<i64> {
        match self {
            ColumnData::I64(v) => Some(v[idx]),
            ColumnData::I32(v) => Some(v[idx] as i64),
            ColumnData::I16(v) => Some(v[idx] as i64),
            ColumnData::I8(v) => Some(v[idx] as i64),
            ColumnData::U64(v) => Some(v[idx] as i64),
            ColumnData::U32(v) => Some(v[idx] as i64),
            ColumnData::U16(v) => Some(v[idx] as i64),
            ColumnData::U8(v) => Some(v[idx] as i64),
            ColumnData::F32(v) => Some(v[idx] as i64),
            ColumnData::F64(v) => Some(v[idx] as i64),
            ColumnData::Bool(v) => Some(i64::from(v[idx])),
        }
    }

    pub(crate) fn is_float(&self) -> bool {
        matches!(self, ColumnData::F32(_) | ColumnData::F64(_))
    }

    pub(crate) fn into_series(self, name: &str) -> Series {
        match self {
            ColumnData::Bool(v) => Series::new(name.into(), v),
            ColumnData::I8(v) => Series::new(name.into(), v),
            ColumnData::I16(v) => Series::new(name.into(), v),
            ColumnData::I32(v) => Series::new(name.into(), v),
            ColumnData::I64(v) => Series::new(name.into(), v),
            ColumnData::U8(v) => Series::new(name.into(), v),
            ColumnData::U16(v) => Series::new(name.into(), v),
            ColumnData::U32(v) => Series::new(name.into(), v),
            ColumnData::U64(v) => Series::new(name.into(), v),
            ColumnData::F32(v) => Series::new(name.into(), v),
            ColumnData::F64(v) => Series::new(name.into(), v),
        }
    }
}

