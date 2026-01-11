// Split from the historical `src/zarr_source.rs` mega-file.
//
// We keep *one* `#[pymethods]` block (PyO3 requires this) and move the heavy
// implementations into plain `impl ZarrSource` helpers.

include!("zarr_source.rs");
include!("zarr_source_new.rs");
include!("zarr_source_predicate.rs");
include!("zarr_source_next.rs");
include!("zarr_source_pymethods.rs");
