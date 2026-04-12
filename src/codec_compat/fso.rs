//! Fixed-scale-offset codec construction with NumPy-style dtype normalization.

use std::sync::Arc;

use zarrs::array::codec::{
    FixedScaleOffsetCodec,
    FixedScaleOffsetCodecConfiguration,
};
use zarrs::metadata::v2::MetadataV2;
use zarrs::metadata::v3::MetadataV3;
use zarrs::plugin::PluginCreateError;
use zarrs_codec::Codec;

use super::dtype::normalize_fixedscaleoffset_dtype_str;

fn apply_fixedscaleoffset_numpy_dtype_normalization(
    configuration: &mut FixedScaleOffsetCodecConfiguration,
) -> Result<(), PluginCreateError> {
    if let FixedScaleOffsetCodecConfiguration::Numcodecs(n) =
        configuration
    {
        n.dtype =
            normalize_fixedscaleoffset_dtype_str(&n.dtype);
        if let Some(a) = n.astype.as_mut() {
            *a = normalize_fixedscaleoffset_dtype_str(a);
        }
        Ok(())
    } else {
        Err(PluginCreateError::Other(
            "unsupported fixedscaleoffset configuration variant".to_string(),
        ))
    }
}

pub(crate) fn fixedscaleoffset_inner_from_v3(
    metadata: &MetadataV3,
) -> Result<
    FixedScaleOffsetCodec,
    PluginCreateError,
> {
    let mut configuration: FixedScaleOffsetCodecConfiguration =
        metadata.to_typed_configuration()?;
    apply_fixedscaleoffset_numpy_dtype_normalization(
        &mut configuration,
    )?;
    FixedScaleOffsetCodec::new_with_configuration(
        &configuration,
    )
}

pub(crate) fn fixedscaleoffset_inner_from_v2(
    metadata: &MetadataV2,
) -> Result<
    FixedScaleOffsetCodec,
    PluginCreateError,
> {
    let mut configuration: FixedScaleOffsetCodecConfiguration =
        metadata.to_typed_configuration()?;
    apply_fixedscaleoffset_numpy_dtype_normalization(
        &mut configuration,
    )?;
    FixedScaleOffsetCodec::new_with_configuration(
        &configuration,
    )
}

/// V3 plugin body: normalized FSO, exposed for the canonical runtime registration.
pub(crate) fn fixedscaleoffset_codec_from_v3_metadata(
    metadata: &MetadataV3,
) -> Result<Codec, PluginCreateError> {
    let inner =
        fixedscaleoffset_inner_from_v3(metadata)?;
    Ok(Codec::ArrayToArray(Arc::new(inner)))
}

/// V2 plugin body.
pub(crate) fn fixedscaleoffset_codec_from_v2_metadata(
    metadata: &MetadataV2,
) -> Result<Codec, PluginCreateError> {
    let inner =
        fixedscaleoffset_inner_from_v2(metadata)?;
    Ok(Codec::ArrayToArray(Arc::new(inner)))
}
