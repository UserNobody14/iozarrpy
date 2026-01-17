// Split from the historical `src/zarr_scan_async.rs` mega-file.

mod prelude;
mod open_arrays;
mod chunk_to_df;
mod scan_async;

pub(crate) use scan_async::scan_zarr_df_async;
