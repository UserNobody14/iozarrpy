use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use polars::prelude::{DataType as PlDataType, Field, Schema};
use smallvec::SmallVec;

use crate::{IStr, IntoIStr};

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

