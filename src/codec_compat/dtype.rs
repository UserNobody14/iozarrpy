//! Map numcodecs / NumPy-style dtype labels in fixed-scale-offset codec config to Zarr V2
//! short names zarrs accepts (`<u2`, `|u1`, …).

/// Map NumPy / xarray-style dtype labels to Zarr V2 short names that zarrs registers
/// (`<u2`, `|u1`, …). Endianness from an existing prefix is preserved (except `u1` / `i1`,
/// which are always normalized to bare `u1` / `i1` so zarrs' `add_byteoder_to_dtype` applies).
pub(crate) fn normalize_fixedscaleoffset_dtype_str(s: &str) -> String {
    let t = s.trim();
    let (endian, core) = split_endian_prefix(t);
    let Some(short) = map_numpy_core_to_short(core) else {
        return t.to_string();
    };

    if short == "u1" {
        return "u1".to_string();
    }
    if short == "i1" {
        // zarrs 0.23 `add_byteoder_to_dtype` does not treat `i1` like `u1` (`|i1`); int8 packed
        // arrays may still fail until upstream fixes that path.
        return "i1".to_string();
    }

    format!("{endian}{short}")
}

fn split_endian_prefix(s: &str) -> (&str, &str) {
    if let Some(rest) = s.strip_prefix('<') {
        ("<", rest)
    } else if let Some(rest) = s.strip_prefix('>') {
        (">", rest)
    } else if let Some(rest) = s.strip_prefix('|') {
        ("|", rest)
    } else {
        ("<", s)
    }
}

fn map_numpy_core_to_short(core: &str) -> Option<&'static str> {
    Some(match core {
        "uint8" | "u1" => "u1",
        "int8" | "i1" => "i1",
        "uint16" | "u2" | "ushort" | "H" => "u2",
        "int16" | "i2" | "short" | "h" => "i2",
        "uint32" | "u4" | "uintc" | "I" => "u4",
        "int32" | "i4" | "intc" | "i" => "i4",
        "uint64" | "u8" | "ulonglong" | "Q" => "u8",
        "int64" | "i8" | "longlong" | "q" => "i8",
        "float16" | "f2" | "half" | "e" => "f2",
        "float32" | "f4" | "single" | "f" => "f4",
        "float64" | "f8" | "double" | "d" => "f8",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint16_to_u2() {
        assert_eq!(
            normalize_fixedscaleoffset_dtype_str("uint16"),
            "<u2"
        );
        assert_eq!(
            normalize_fixedscaleoffset_dtype_str(">uint16"),
            ">u2"
        );
    }

    #[test]
    fn uint8_to_u1_token() {
        assert_eq!(normalize_fixedscaleoffset_dtype_str("uint8"), "u1");
    }
}
