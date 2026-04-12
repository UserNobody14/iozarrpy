//! One-time registration of runtime zarrs codec plugins.
//!
//! zarrs checks the runtime registry before compile-time `inventory` plugins
//! ([`Codec::from_metadata`](zarrs_codec::Codec::from_metadata)). See
//! [zarrs codec extensions](https://book.zarrs.dev/extensions/codec.html).

use std::sync::Once;

use zarrs_codec::{
    CodecRuntimePluginV2, CodecRuntimePluginV3, register_codec_v2,
    register_codec_v3,
};

use super::aliases::ensure_alias_codec_plugin_registered;
use super::fso::{
    fixedscaleoffset_codec_from_v2_metadata,
    fixedscaleoffset_codec_from_v3_metadata,
};

/// Register runtime codec plugins (vendor alias table + normalized fixed-scale-offset).
///
/// Safe to call from multiple threads; subsequent calls are no-ops.
pub(crate) fn ensure_zarr_compat_registered() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        ensure_alias_codec_plugin_registered();

        let _v3 = register_codec_v3(CodecRuntimePluginV3::new(
            |name| name == "numcodecs.fixedscaleoffset",
            fixedscaleoffset_codec_from_v3_metadata,
        ));
        let _v2 = register_codec_v2(CodecRuntimePluginV2::new(
            |id| id == "fixedscaleoffset",
            fixedscaleoffset_codec_from_v2_metadata,
        ));
    });
}
