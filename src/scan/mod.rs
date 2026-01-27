// Split from the historical `src/zarr_scan_async.rs` mega-file.

mod prelude;
pub(crate) mod open_arrays;
pub(crate) mod chunk_to_df;
mod scan_async;
pub(crate) mod grid_combiner;

pub(crate) use scan_async::scan_zarr_df_async;
