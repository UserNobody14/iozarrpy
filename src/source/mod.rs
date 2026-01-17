// Split from the historical `src/zarr_source.rs` mega-file.
//
// We keep *one* `#[pymethods]` block (PyO3 requires this) and move the heavy
// implementations into plain `impl ZarrSource` helpers.

mod zarr_source;

pub use zarr_source::ZarrSource;
