use std::collections::{BTreeMap, BTreeSet};

use polars::prelude::{DataType as PlDataType, Field, Schema, TimeUnit};
use zarrs::array::Array;
use zarrs::hierarchy::NodeMetadata;

use crate::zarr_store::{open_store, OpenedStore};

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
    /// Decode a raw zarr value to nanoseconds since Unix epoch.
    #[inline]
    pub fn decode(&self, raw: i64) -> i64 {
        if self.is_duration {
            raw * self.unit_ns
        } else {
            self.epoch_ns + raw * self.unit_ns
        }
    }

    /// Encode a nanoseconds value back to the raw zarr representation.
    #[inline]
    pub fn encode(&self, ns: i64) -> i64 {
        if self.is_duration {
            ns / self.unit_ns
        } else {
            (ns - self.epoch_ns) / self.unit_ns
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZarrArrayMeta {
    pub name: String,
    pub path: String,
    pub shape: Vec<u64>,
    pub dims: Vec<String>,
    pub zarr_dtype: String,
    pub polars_dtype: PlDataType,
    /// Optional time encoding if this array represents datetime or duration.
    pub time_encoding: Option<TimeEncoding>,
}

#[derive(Debug, Clone)]
pub struct ZarrDatasetMeta {
    pub root: String,
    /// Arrays indexed by their resolved column name (currently leaf name, de-duped if needed).
    pub arrays: BTreeMap<String, ZarrArrayMeta>,
    pub dims: Vec<String>,
    pub coords: Vec<String>,
    pub data_vars: Vec<String>,
}

impl ZarrDatasetMeta {
    /// Build a “tidy table” schema: coord columns + variable columns.
    ///
    /// If a dimension has a matching 1D coordinate array (same name), the coord column uses that dtype.
    /// Otherwise the coord column is an `Int64` index.
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
            Box::new(self.data_vars.iter().map(|s| s.as_str()).filter(|v| var_set.contains(v)))
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

pub fn load_dataset_meta(zarr_url: &str) -> Result<ZarrDatasetMeta, String> {
    let OpenedStore { store, root } = open_store(zarr_url)?;
    load_dataset_meta_from_opened(&OpenedStore { store, root })
}

pub fn load_dataset_meta_from_opened(opened: &OpenedStore) -> Result<ZarrDatasetMeta, String> {
    let store = opened.store.clone();
    let root = opened.root.clone();

    // Open the root group and traverse nodes.
    let group = zarrs::group::Group::open(store.clone(), &root).map_err(to_string_err)?;
    let nodes = group.traverse().map_err(to_string_err)?;

    let mut arrays: BTreeMap<String, ZarrArrayMeta> = BTreeMap::new();
    let mut seen_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut dims_seen: BTreeSet<String> = BTreeSet::new();
    let mut dims_ordered: Vec<String> = Vec::new();
    let mut coord_candidates: BTreeMap<String, (Vec<u64>, PlDataType)> = BTreeMap::new();
    // Track the array with the most dimensions to use as the primary dimension ordering
    let mut primary_dims: Option<Vec<String>> = None;
    let mut max_ndim = 0usize;

    for (path, md) in nodes {
        if !matches!(md, NodeMetadata::Array(_)) {
            continue;
        }

        let path_str = path.as_str().to_string();
        let leaf = leaf_name(&path_str);

        let array = Array::open(store.clone(), &path_str).map_err(to_string_err)?;

        let shape = array.shape().to_vec();
        let dims = dims_for_array(&array).unwrap_or_else(|| default_dims(shape.len()));
        
        // Track dims in order of first appearance, and keep the longest dim list
        for d in &dims {
            if dims_seen.insert(d.clone()) {
                dims_ordered.push(d.clone());
            }
        }
        // Prefer dimension order from the array with most dimensions (likely the main data var)
        if dims.len() > max_ndim {
            max_ndim = dims.len();
            primary_dims = Some(dims.clone());
        }

        // Extract time encoding if present
        let time_encoding = extract_time_encoding(&array);

        // Candidate coord: 1D array whose leaf name equals its (only) dim name.
        if shape.len() == 1 && dims.len() == 1 && leaf == dims[0] {
            let dt = zarr_dtype_to_polars(array.data_type().identifier(), time_encoding.as_ref());
            coord_candidates.insert(leaf.clone(), (shape.clone(), dt));
        }

        let zarr_dtype = array.data_type().identifier().to_string();
        let polars_dtype = zarr_dtype_to_polars(&zarr_dtype, time_encoding.as_ref());

        // De-dupe by leaf name if collisions happen (nested groups).
        let name = match seen_names.get_mut(&leaf) {
            None => {
                seen_names.insert(leaf.clone(), 1);
                leaf.clone()
            }
            Some(n) => {
                *n += 1;
                format!("{leaf}__{n}")
            }
        };

        arrays.insert(
            name.clone(),
            ZarrArrayMeta {
                name,
                path: path_str,
                shape,
                dims,
                zarr_dtype,
                polars_dtype,
                time_encoding,
            },
        );
    }

    // Use dimension order from the array with most dimensions (main data variable).
    // This preserves the intended order (e.g., time, lead_time, y, x) rather than alphabetical.
    let dims: Vec<String> = primary_dims.unwrap_or(dims_ordered);

    // Determine coords: any dim that has a matching 1D coordinate array.
    let mut coords: Vec<String> = Vec::new();
    for dim in &dims {
        if let Some((shape, _dtype)) = coord_candidates.get(dim) {
            // Must match dimension length (if dim length is known from any data var).
            // We keep it simple: accept any 1D coord; later we’ll validate against vars.
            if shape.len() == 1 {
                coords.push(dim.clone());
            }
        }
    }

    // Determine data variables: arrays that are not classified as coords by name.
    let coord_set: BTreeSet<&str> = coords.iter().map(|s| s.as_str()).collect();
    let data_vars: Vec<String> = arrays
        .keys()
        .filter(|k| !coord_set.contains(k.as_str()))
        .cloned()
        .collect();

    Ok(ZarrDatasetMeta {
        root,
        arrays,
        dims,
        coords,
        data_vars,
    })
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn leaf_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or_default().to_string()
}

fn default_dims(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("dim_{i}")).collect()
}

fn dims_for_array<TStorage: ?Sized>(array: &Array<TStorage>) -> Option<Vec<String>> {
    // Prefer xarray-style attribute, when present.
    if let Some(v) = array.attributes().get("_ARRAY_DIMENSIONS") {
        if let Some(list) = v.as_array() {
            let out: Vec<String> = list
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect();
            if !out.is_empty() {
                return Some(out);
            }
        }
    }

    // Fall back to Zarr V3 dimension_names.
    if let Some(names) = array.dimension_names() {
        let out: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(i, n)| n.clone().unwrap_or_else(|| format!("dim_{i}")))
            .collect();
        return Some(out);
    }

    None
}

fn zarr_dtype_to_polars(zarr_identifier: &str, time_encoding: Option<&TimeEncoding>) -> PlDataType {
    // If we have time encoding, return the appropriate temporal type.
    if let Some(te) = time_encoding {
        return if te.is_duration {
            PlDataType::Duration(TimeUnit::Nanoseconds)
        } else {
            PlDataType::Datetime(TimeUnit::Nanoseconds, None)
        };
    }

    // Conservative, first-milestone mapping.
    match zarr_identifier {
        "bool" => PlDataType::Boolean,
        "int8" => PlDataType::Int8,
        "int16" => PlDataType::Int16,
        "int32" => PlDataType::Int32,
        "int64" => PlDataType::Int64,
        "uint8" => PlDataType::UInt8,
        "uint16" => PlDataType::UInt16,
        "uint32" => PlDataType::UInt32,
        "uint64" => PlDataType::UInt64,
        "float16" | "bfloat16" => PlDataType::Float32,
        "float32" => PlDataType::Float32,
        "float64" => PlDataType::Float64,
        "string" => PlDataType::String,
        // Keep unknowns representable; we can error later in the reader if needed.
        _ => PlDataType::Binary,
    }
}

/// Parse CF-conventions time units string like "hours since 2024-01-01 00:00:00".
/// Returns (unit_ns, epoch_ns) or None if parsing fails.
fn parse_cf_time_units(units: &str) -> Option<(i64, i64)> {
    // Format: "<unit> since <datetime>"
    let parts: Vec<&str> = units.splitn(2, " since ").collect();
    if parts.len() != 2 {
        return None;
    }

    let unit_str = parts[0].trim().to_lowercase();
    let epoch_str = parts[1].trim();

    // Map time unit to nanoseconds
    let unit_ns: i64 = match unit_str.as_str() {
        "nanoseconds" | "nanosecond" | "ns" => 1,
        "microseconds" | "microsecond" | "us" | "µs" => 1_000,
        "milliseconds" | "millisecond" | "ms" => 1_000_000,
        "seconds" | "second" | "s" => 1_000_000_000,
        "minutes" | "minute" | "min" => 60 * 1_000_000_000,
        "hours" | "hour" | "h" | "hr" => 3600 * 1_000_000_000,
        "days" | "day" | "d" => 86400 * 1_000_000_000,
        _ => return None,
    };

    // Parse the epoch datetime
    // Try common formats: "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DD"
    let epoch_ns = parse_datetime_to_ns(epoch_str)?;

    Some((unit_ns, epoch_ns))
}

/// Parse a datetime string to nanoseconds since Unix epoch.
fn parse_datetime_to_ns(s: &str) -> Option<i64> {
    use chrono::{NaiveDateTime, NaiveDate, TimeZone, Utc};

    // Try various formats
    // Format: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS.fff"
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?);
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?);
    }
    // Format: "YYYY-MM-DDTHH:MM:SS"
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?);
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?);
    }
    // Format: "YYYY-MM-DD"
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = d.and_hms_opt(0, 0, 0)?;
        return Some(Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?);
    }

    None
}

/// Parse xarray's dtype attribute for timedelta (e.g., "timedelta64[ns]").
/// Returns unit_ns or None.
fn parse_timedelta_dtype(dtype_str: &str) -> Option<i64> {
    // Format: "timedelta64[<unit>]"
    if !dtype_str.starts_with("timedelta64[") || !dtype_str.ends_with(']') {
        return None;
    }
    let unit = &dtype_str[12..dtype_str.len() - 1];
    match unit {
        "ns" => Some(1),
        "us" => Some(1_000),
        "ms" => Some(1_000_000),
        "s" => Some(1_000_000_000),
        "m" => Some(60 * 1_000_000_000),
        "h" => Some(3600 * 1_000_000_000),
        "D" => Some(86400 * 1_000_000_000),
        _ => None,
    }
}

/// Parse duration units string (just the unit, no "since" part) for timedelta arrays.
fn parse_duration_units(units: &str) -> Option<i64> {
    let unit_str = units.trim().to_lowercase();
    match unit_str.as_str() {
        "nanoseconds" | "nanosecond" | "ns" => Some(1),
        "microseconds" | "microsecond" | "us" | "µs" => Some(1_000),
        "milliseconds" | "millisecond" | "ms" => Some(1_000_000),
        "seconds" | "second" | "s" => Some(1_000_000_000),
        "minutes" | "minute" | "min" => Some(60 * 1_000_000_000),
        "hours" | "hour" | "h" | "hr" => Some(3600 * 1_000_000_000),
        "days" | "day" | "d" => Some(86400 * 1_000_000_000),
        _ => None,
    }
}

/// Extract time encoding from array attributes (CF conventions).
fn extract_time_encoding<TStorage: ?Sized>(array: &Array<TStorage>) -> Option<TimeEncoding> {
    let attrs = array.attributes();

    // Check for xarray's dtype attribute first (timedelta64)
    if let Some(dtype_val) = attrs.get("dtype") {
        if let Some(dtype_str) = dtype_val.as_str() {
            if let Some(dtype_unit_ns) = parse_timedelta_dtype(dtype_str) {
                // This is a timedelta/duration array
                // Check for units attribute to get the storage unit multiplier
                let storage_unit_ns = attrs
                    .get("units")
                    .and_then(|v| v.as_str())
                    .and_then(parse_duration_units)
                    .unwrap_or(dtype_unit_ns);

                return Some(TimeEncoding {
                    epoch_ns: 0,
                    unit_ns: storage_unit_ns,
                    is_duration: true,
                });
            }
        }
    }

    // Check for CF datetime encoding: "units" with "since" and optional "calendar"
    let units = attrs.get("units").and_then(|v| v.as_str())?;

    // If it contains "since", it's a datetime encoding
    if units.contains(" since ") {
        let (unit_ns, epoch_ns) = parse_cf_time_units(units)?;
        return Some(TimeEncoding {
            epoch_ns,
            unit_ns,
            is_duration: false,
        });
    }

    None
}

