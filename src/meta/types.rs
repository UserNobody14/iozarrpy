use std::collections::{BTreeMap, BTreeSet};

use polars::prelude::{DataType as PlDataType, Field, Schema};

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
    pub path: String,
    pub shape: Vec<u64>,
    pub dims: Vec<String>,
    pub polars_dtype: PlDataType,
    pub time_encoding: Option<TimeEncoding>,
}

#[derive(Debug, Clone)]
pub struct ZarrDatasetMeta {
    pub arrays: BTreeMap<String, ZarrArrayMeta>,
    pub dims: Vec<String>,
    pub data_vars: Vec<String>,
}

impl ZarrDatasetMeta {
    pub fn tidy_schema(&self, variables: Option<&[String]>) -> Schema {
        let var_set: Option<BTreeSet<&str>> = variables.map(|v| v.iter().map(|s| s.as_str()).collect());

        let mut fields: Vec<Field> = Vec::new();

        for dim in &self.dims {
            let dtype = self
                .arrays
                .get(dim)
                .map(|m| m.polars_dtype.clone())
                .unwrap_or(PlDataType::Int64);
            fields.push(Field::new(dim.into(), dtype));
        }

        let vars_iter: Box<dyn Iterator<Item = &str>> = if let Some(var_set) = &var_set {
            Box::new(
                self.data_vars
                    .iter()
                    .map(|s| s.as_str())
                    .filter(|v| var_set.contains(v)),
            )
        } else {
            Box::new(self.data_vars.iter().map(|s| s.as_str()))
        };

        for v in vars_iter {
            if let Some(m) = self.arrays.get(v) {
                fields.push(Field::new(v.into(), m.polars_dtype.clone()));
            }
        }

        fields.into_iter().collect()
    }
}

