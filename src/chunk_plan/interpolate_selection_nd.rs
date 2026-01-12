use std::collections::HashMap;

use polars::prelude::AnyValue;

use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::selection::DatasetSelection;
use super::prelude::*;

pub(super) fn interpolate_selection_nd(source_coords: Expr, source_values: Expr, target_scheme: Expr, kwargs: &Arc<[u8]>, ctx: &mut CompileCtx<'_>) -> Result<DatasetSelection, CompileError> {
    Ok(ctx.all())
}