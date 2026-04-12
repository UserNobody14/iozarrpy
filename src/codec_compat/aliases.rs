//! User-configurable Zarr V3 codec name aliases.
//!
//! Some stores use vendor-prefixed codec names in metadata. Map those to zarrs canonical names
//! (`numcodecs.bitround`, `numcodecs.fixedscaleoffset`). Fixed-scale-offset uses
//! [`super::fso`] for NumPy-style dtype normalization.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Once;

use parking_lot::RwLock;
use zarrs::array::codec::{
    BitroundCodec, BitroundCodecConfiguration,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::plugin::PluginCreateError;
use zarrs_codec::{
    Codec, CodecRuntimePluginV3,
    register_codec_v3,
};

use super::fso::fixedscaleoffset_inner_from_v3;

static ALIASES: LazyLock<
    RwLock<HashMap<String, String>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));

static ALIAS_PLUGIN: Once = Once::new();

fn resolve_canonical_target(
    raw: &str,
) -> Result<&'static str, String> {
    match raw.trim() {
        "numcodecs.bitround" | "bitround" => {
            Ok("numcodecs.bitround")
        }
        "numcodecs.fixedscaleoffset"
        | "fixedscaleoffset" => {
            Ok("numcodecs.fixedscaleoffset")
        }
        other => Err(format!(
            "unknown codec alias target {other:?}; expected one of: \
             \"numcodecs.bitround\", \"bitround\", \
             \"numcodecs.fixedscaleoffset\", \"fixedscaleoffset\""
        )),
    }
}

fn dispatch_aliased_v3(
    metadata: &MetadataV3,
    canonical: &str,
) -> Result<Codec, PluginCreateError> {
    match canonical {
        "numcodecs.bitround" => {
            let configuration: BitroundCodecConfiguration =
                metadata.to_typed_configuration()?;
            let inner = BitroundCodec::new_with_configuration(
                &configuration,
            )?;
            Ok(Codec::ArrayToArray(Arc::new(
                inner,
            )))
        }
        "numcodecs.fixedscaleoffset" => {
            let inner =
                fixedscaleoffset_inner_from_v3(
                    metadata,
                )?;
            Ok(Codec::ArrayToArray(Arc::new(
                inner,
            )))
        }
        _ => Err(PluginCreateError::Other(
            format!(
                "internal error: unhandled canonical codec {canonical:?}"
            ),
        )),
    }
}

/// Replace the global V3 codec alias table. Keys are names as they appear in array metadata;
/// values are canonical codec names (see [`resolve_canonical_target`]).
pub fn set_codec_aliases(
    aliases: HashMap<String, String>,
) -> Result<(), String> {
    let mut resolved: HashMap<String, String> =
        HashMap::with_capacity(aliases.len());
    for (from, to) in aliases {
        if from.trim().is_empty() {
            return Err("codec alias key must not be empty".to_string());
        }
        let to_resolved =
            resolve_canonical_target(&to)?
                .to_string();
        resolved.insert(from, to_resolved);
    }
    *ALIASES.write() = resolved;
    Ok(())
}

fn alias_match_name(name: &str) -> bool {
    ALIASES.read().contains_key(name)
}

/// Register a single runtime plugin that consults [`ALIASES`] on each resolve.
pub(crate) fn ensure_alias_codec_plugin_registered()
 {
    ALIAS_PLUGIN.call_once(|| {
        let _h = register_codec_v3(CodecRuntimePluginV3::new(
            alias_match_name,
            |metadata: &MetadataV3| {
                let canonical = ALIASES
                    .read()
                    .get(metadata.name())
                    .cloned()
                    .ok_or_else(|| {
                        PluginCreateError::Other(
                            "codec alias missing for name (race?)".to_string(),
                        )
                    })?;
                dispatch_aliased_v3(metadata, &canonical)
            },
        ));
    });
}
