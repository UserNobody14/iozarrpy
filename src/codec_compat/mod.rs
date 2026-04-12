//! Zarr codec compatibility: runtime plugin registration, vendor codec aliases, and
//! numcodecs fixed-scale-offset dtype normalization for zarrs.

mod aliases;
mod dtype;
mod fso;
mod runtime;

pub use aliases::set_codec_aliases;

pub(crate) use runtime::ensure_zarr_compat_registered;
