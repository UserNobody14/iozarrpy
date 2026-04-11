//! Interpolation selection compilation (interpolate_nd / interpolate_geospatial FfiPlugin).

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::{ExprPlan, VarSet};
use super::super::expr_utils::{
    extract_column_names_lazy,
    extract_literal_struct_series_lazy,
    series_values_scalar_lazy,
};
use super::super::literals::strip_wrappers;
use crate::chunk_plan::exprs::compile::expr::compile_expr_list;
use crate::chunk_plan::exprs::compile::utils::collect_refs_from_expr_list;
use crate::{try_extract};
use crate::chunk_plan::indexing::lazy_selection::{
    LazyArraySelection, LazyDimConstraint, LazyHyperRectangle,
};
use crate::chunk_plan::indexing::selection::SetOperations;
use crate::chunk_plan::indexing::types::{CoordScalar, ValueRangePresent};
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::{IStr, IntoIStr};

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile interpolation selection (lazy version).
pub(super) fn interpolate_selection_nd_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    try_extract!(let Some(coord_names) = extract_column_names_lazy(source_coords));
    try_extract!(let Some(target_struct) =
        extract_literal_struct_series_lazy(
            target_values,
        )
    );
    try_extract!(let Ok(target_sc) = target_struct.struct_());
    let target_fields =
        target_sc.fields_as_series();

    let mut dim_values: std::collections::BTreeMap<IStr, Vec<CoordScalar>> =
        std::collections::BTreeMap::new();

    // Include all target dimensions that match ctx.dims, not just coord_names.
    // Extra dims (e.g. time in target when interpolating lat/lon) must be
    // constrained to exact match so we load only the relevant slice.
    for s in target_fields.iter() {
        let name = s.name().as_str().istr();
        if !ctx.dims.iter().any(|d| d == &name) {
            continue;
        }

        try_extract!(let Some(values) = series_values_scalar_lazy(s));

        if !values.is_empty() {
            dim_values.insert(name, values);
        }
    }

    if dim_values.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let mut constraints: Vec<
        std::collections::BTreeMap<
            IStr,
            LazyDimConstraint,
        >,
    > = vec![];

    // Transform dim_values (dim -> Vec<CoordScalar>) to row-wise constraints.
    // Use InterpolationRange (with expansion) only for coord_names; use Unresolved
    // (no expansion) for filter dimensions so we load only the exact slice.
    let num_rows = dim_values
        .values()
        .next()
        .map(|v| v.len())
        .unwrap_or(0);
    for i in 0..num_rows {
        let mut constraint =
            std::collections::BTreeMap::new();
        for (dim_name, values) in
            dim_values.iter()
        {
            let value = values[i].clone();
            let vr = ValueRangePresent::from_equal_case(value);
            let is_interp_dim = coord_names
                .iter()
                .any(|c| c == dim_name);
            let c = if is_interp_dim {
                LazyDimConstraint::InterpolationRange(vr)
            } else {
                LazyDimConstraint::Unresolved(vr)
            };
            constraint
                .insert(*dim_name, c);
        }
        constraints.push(constraint);
    }

    let rects: Vec<LazyHyperRectangle> =
        constraints
            .into_iter()
            .map(|c| {
                LazyHyperRectangle::with_dims(c)
            })
            .collect();
    let sel = LazyArraySelection::Rectangles(
        rects.into(),
    );

    let (retrieve_vars, filter_plan) =
        match source_values {
            Expr::Function {
                input,
                function,
            } => match function {
                FunctionExpr::AsStruct => {
                    let filter_initial =
                        compile_expr_list(
                            input, ctx,
                        )?;
                    let vars =
                    collect_refs_from_expr_list(
                            input,
                        );
                    (vars, Some(filter_initial))
                }
                _ => {
                    return Err(BackendError::compile_polars(format!(
                    "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                     containing column refs or col(...).filter(predicate): {:?}",
                    source_values
                )));
                }
            },
            Expr::Field(names) => (
                names
                    .iter()
                    .map(|n| n.istr())
                    .collect::<Vec<_>>(),
                None,
            ),
            _ => {
                return Err(
                    BackendError::compile_polars(
                        format!(
                            "source_values must be an Expr::Field or AsStruct containing variable names: {:?}",
                            source_values
                        ),
                    ),
                );
            }
        };

    let sel = match filter_plan {
        Some(ExprPlan::Active {
            constraints: filter_sel,
            ..
        }) => sel.intersect(filter_sel.as_ref()),
        Some(ExprPlan::Empty) => {
            return Ok(ExprPlan::Empty);
        }
        Some(ExprPlan::NoConstraint) | None => {
            sel
        }
    };

    if sel.is_empty() {
        return Ok(ExprPlan::Empty);
    }

    Ok(ExprPlan::constrained(
        VarSet::from_vec(retrieve_vars),
        sel,
    ))
}

/// Extract coordinate column names from an AsStruct expression, preserving
/// input order. For `pl.struct([source_lat, source_lon])` this yields
/// `["lat", "lon"]` — the last element is always longitude.
fn extract_coord_names_ordered(
    expr: &Expr,
) -> Option<Vec<IStr>> {
    let expr = strip_wrappers(expr);
    if let Expr::Function { input, function } =
        expr
        && matches!(
            function,
            FunctionExpr::AsStruct
        ) {
            let mut names =
                Vec::with_capacity(input.len());
            for e in input {
                let mut found = None;
                walk_for_first_column(
                    strip_wrappers(e),
                    &mut found,
                );
                names.push(found?);
            }
            return if names.is_empty() {
                None
            } else {
                Some(names)
            };
        }
    None
}

fn walk_for_first_column(
    expr: &Expr,
    out: &mut Option<IStr>,
) {
    if out.is_some() {
        return;
    }
    match expr {
        Expr::Column(name) => {
            *out = Some(name.istr());
        }
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Sort { expr: inner, .. } => {
            walk_for_first_column(inner, out);
        }
        _ => {}
    }
}

/// Compile geospatial interpolation selection (lazy version).
///
/// Like `interpolate_selection_nd_lazy` but uses `WrappingInterpolationRange`
/// for the longitude dimension (the last coordinate) to handle ghost-point
/// expansion for periodic grids.
pub(super) fn interpolate_selection_geospatial_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let coord_names_ordered =
        extract_coord_names_ordered(
            source_coords,
        );
    let coord_names_flat = coord_names_ordered
        .as_deref()
        .or({
            // Fallback: unordered extraction (can't distinguish lon)
            None
        });

    let coord_names: Vec<IStr> =
        match coord_names_flat {
            Some(names) => names.to_vec(),
            None => {
                try_extract!(let Some(names) = extract_column_names_lazy(source_coords));
                names
            }
        };

    let lon_dim: Option<&IStr> =
        coord_names_ordered
            .as_ref()
            .and_then(|names| names.last());

    try_extract!(let Some(target_struct) =
        extract_literal_struct_series_lazy(target_values)
    );
    try_extract!(let Ok(target_sc) = target_struct.struct_());
    let target_fields =
        target_sc.fields_as_series();

    let mut dim_values: std::collections::BTreeMap<IStr, Vec<CoordScalar>> =
        std::collections::BTreeMap::new();

    for s in target_fields.iter() {
        let name = s.name().as_str().istr();
        if !ctx.dims.iter().any(|d| d == &name) {
            continue;
        }

        try_extract!(let Some(values) = series_values_scalar_lazy(s));

        if !values.is_empty() {
            dim_values.insert(name, values);
        }
    }

    if dim_values.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let mut constraints: Vec<
        std::collections::BTreeMap<
            IStr,
            LazyDimConstraint,
        >,
    > = vec![];

    let num_rows = dim_values
        .values()
        .next()
        .map(|v| v.len())
        .unwrap_or(0);
    for i in 0..num_rows {
        let mut constraint =
            std::collections::BTreeMap::new();
        for (dim_name, values) in
            dim_values.iter()
        {
            let value = values[i].clone();
            let vr = ValueRangePresent::from_equal_case(value);
            let is_interp_dim = coord_names
                .iter()
                .any(|c| c == dim_name);
            let is_lon = lon_dim == Some(dim_name);
            let c = if is_lon {
                LazyDimConstraint::WrappingInterpolationRange(vr)
            } else if is_interp_dim {
                LazyDimConstraint::InterpolationRange(vr)
            } else {
                LazyDimConstraint::Unresolved(vr)
            };
            constraint
                .insert(*dim_name, c);
        }
        constraints.push(constraint);
    }

    let rects: Vec<LazyHyperRectangle> =
        constraints
            .into_iter()
            .map(|c| {
                LazyHyperRectangle::with_dims(c)
            })
            .collect();
    let sel = LazyArraySelection::Rectangles(
        rects.into(),
    );

    let (retrieve_vars, filter_plan) =
        match source_values {
            Expr::Function {
                input,
                function,
            } => match function {
                FunctionExpr::AsStruct => {
                    let filter_initial =
                        compile_expr_list(
                            input, ctx,
                        )?;
                    let vars =
                        collect_refs_from_expr_list(input);
                    (vars, Some(filter_initial))
                }
                _ => {
                    return Err(BackendError::compile_polars(format!(
                        "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                         containing column refs or col(...).filter(predicate): {:?}",
                        source_values
                    )));
                }
            },
            Expr::Field(names) => (
                names
                    .iter()
                    .map(|n| n.istr())
                    .collect::<Vec<_>>(),
                None,
            ),
            _ => {
                return Err(
                    BackendError::compile_polars(
                        format!(
                            "source_values must be an Expr::Field or AsStruct containing variable names: {:?}",
                            source_values
                        ),
                    ),
                );
            }
        };

    let sel = match filter_plan {
        Some(ExprPlan::Active {
            constraints: filter_sel,
            ..
        }) => sel.intersect(filter_sel.as_ref()),
        Some(ExprPlan::Empty) => {
            return Ok(ExprPlan::Empty);
        }
        Some(ExprPlan::NoConstraint) | None => {
            sel
        }
    };

    if sel.is_empty() {
        return Ok(ExprPlan::Empty);
    }

    Ok(ExprPlan::constrained(
        VarSet::from_vec(retrieve_vars),
        sel,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_plan::exprs::expr_plan::VarSet;
    use crate::meta::{
        DimensionAnalysis, ZarrArrayMeta,
        ZarrMeta, ZarrNode,
    };
    use polars::prelude::*;
    use smallvec::SmallVec;
    use std::collections::BTreeMap;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use zarrs::array::ChunkGrid;
    use zarrs::array::chunk_grid::regular::RegularChunkGrid;

    fn make_chunk_grid(
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> Arc<ChunkGrid> {
        let cs: Vec<NonZeroU64> = chunk_shape
            .iter()
            .map(|&s| NonZeroU64::new(s).unwrap())
            .collect();
        Arc::new(ChunkGrid::new(
            RegularChunkGrid::new(
                shape.to_vec(),
                cs,
            )
            .unwrap(),
        ))
    }

    fn make_array_meta(
        path: &str,
        dims: &[&str],
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> (IStr, Arc<ZarrArrayMeta>) {
        let dim_sv: SmallVec<[IStr; 4]> = dims
            .iter()
            .map(|d| d.istr())
            .collect();
        let cg =
            make_chunk_grid(shape, chunk_shape);
        let meta = ZarrArrayMeta {
            path: path.istr(),
            shape: shape.into(),
            chunk_shape: chunk_shape.into(),
            chunk_grid: cg,
            dims: dim_sv,
            polars_dtype: DataType::Float64,
            encoding: None,
            array_metadata: None,
        };
        (path.istr(), Arc::new(meta))
    }

    fn make_geo_meta() -> ZarrMeta {
        let mut arrays: BTreeMap<
            IStr,
            Arc<ZarrArrayMeta>,
        > = BTreeMap::new();

        for (name, len, cs) in [
            ("lat", 180u64, 10u64),
            ("lon", 360, 10),
        ] {
            let (key, meta) = make_array_meta(
                name,
                &[name],
                &[len],
                &[cs],
            );
            arrays.insert(key, meta);
        }

        let (key, meta) = make_array_meta(
            "temperature",
            &["lat", "lon"],
            &[180, 360],
            &[10, 10],
        );
        arrays.insert(key, meta);

        let root = ZarrNode {
            path: "/".istr(),
            arrays,
            children: BTreeMap::new(),
            local_dims: vec![
                "lat".istr(),
                "lon".istr(),
            ],
            data_vars: vec!["temperature".istr()],
        };

        let dim_analysis =
            DimensionAnalysis::compute(&root);

        ZarrMeta { root, dim_analysis }
    }

    // =========================================================================
    // extract_coord_names_ordered
    // =========================================================================

    #[test]
    fn ordered_names_from_struct() {
        let struct_expr =
            polars::prelude::as_struct(vec![
                col("lat"),
                col("lon"),
            ]);
        let names = extract_coord_names_ordered(
            &struct_expr,
        );
        assert_eq!(
            names,
            Some(vec![
                "lat".istr(),
                "lon".istr()
            ])
        );
    }

    #[test]
    fn ordered_names_preserves_order() {
        let struct_expr =
            polars::prelude::as_struct(vec![
                col("lon"),
                col("lat"),
            ]);
        let names = extract_coord_names_ordered(
            &struct_expr,
        );
        assert_eq!(
            names,
            Some(vec![
                "lon".istr(),
                "lat".istr()
            ])
        );
    }

    #[test]
    fn ordered_names_non_struct_returns_none() {
        let expr = col("lat");
        let names =
            extract_coord_names_ordered(&expr);
        assert_eq!(names, None);
    }

    #[test]
    fn ordered_names_with_alias() {
        let struct_expr =
            polars::prelude::as_struct(vec![
                col("lat").alias("latitude"),
                col("lon"),
            ]);
        let names = extract_coord_names_ordered(
            &struct_expr,
        );
        assert_eq!(
            names,
            Some(vec![
                "lat".istr(),
                "lon".istr()
            ])
        );
    }

    // =========================================================================
    // interpolate_selection_geospatial_lazy
    // =========================================================================

    fn make_target_struct_lit(
        lats: &[f64],
        lons: &[f64],
    ) -> Expr {
        let lat_col =
            Column::new("lat".into(), lats);
        let lon_col =
            Column::new("lon".into(), lons);
        let target_struct =
            StructChunked::from_columns(
                "__interp_target__".into(),
                lats.len(),
                &[lat_col, lon_col],
            )
            .unwrap()
            .into_series();
        Expr::Literal(LiteralValue::Series(
            polars::prelude::SpecialEq::new(
                target_struct,
            ),
        ))
    }

    fn assert_vars_contain(
        vars: &VarSet,
        name: &str,
    ) {
        match vars {
            VarSet::All => {}
            VarSet::Specific(sv) => {
                assert!(
                    sv.iter().any(|v| {
                        let s: &str = v.as_ref();
                        s == name
                    }),
                    "vars should contain {name}"
                );
            }
        }
    }

    #[test]
    fn geospatial_produces_wrapping_for_lon() {
        let meta = make_geo_meta();
        let dims: Vec<IStr> =
            vec!["lat".istr(), "lon".istr()];
        let mut ctx =
            LazyCompileCtx::new(&meta, &dims);

        let coord_struct =
            polars::prelude::as_struct(vec![
                col("lat"),
                col("lon"),
            ]);
        let value_struct =
            polars::prelude::as_struct(vec![
                col("temperature"),
            ]);
        let target_lit = make_target_struct_lit(
            &[45.0, -30.0],
            &[170.0, -175.0],
        );

        let plan = interpolate_selection_geospatial_lazy(
            &coord_struct,
            &value_struct,
            &target_lit,
            &mut ctx,
        )
        .unwrap();

        match &plan {
            ExprPlan::Active { constraints, .. } => {
                if let LazyArraySelection::Rectangles(rects) = constraints.as_ref() {
                    assert_eq!(rects.len(), 2);
                    for rect in rects.iter() {
                        let mut has_interp = false;
                        let mut has_wrapping = false;
                        for (dim, constraint) in rect.dims() {
                            let dim_str: &str = dim.as_ref();
                            if dim_str == "lat" {
                                assert!(
                                    matches!(constraint, LazyDimConstraint::InterpolationRange(_)),
                                    "lat should use InterpolationRange"
                                );
                                has_interp = true;
                            }
                            if dim_str == "lon" {
                                assert!(
                                    matches!(constraint, LazyDimConstraint::WrappingInterpolationRange(_)),
                                    "lon should use WrappingInterpolationRange"
                                );
                                has_wrapping = true;
                            }
                        }
                        assert!(has_interp, "expected InterpolationRange on lat");
                        assert!(has_wrapping, "expected WrappingInterpolationRange on lon");
                    }
                } else {
                    panic!("expected Rectangles selection");
                }
            }
            other => panic!("expected Active plan, got {:?}", other),
        }
    }

    #[test]
    fn geospatial_no_matching_dims_returns_no_constraint()
     {
        let meta = make_geo_meta();
        let dims: Vec<IStr> = vec!["time".istr()];
        let mut ctx =
            LazyCompileCtx::new(&meta, &dims);

        let coord_struct =
            polars::prelude::as_struct(vec![
                col("lat"),
                col("lon"),
            ]);
        let value_struct =
            polars::prelude::as_struct(vec![
                col("temperature"),
            ]);
        let target_lit = make_target_struct_lit(
            &[45.0],
            &[170.0],
        );

        let plan = interpolate_selection_geospatial_lazy(
            &coord_struct,
            &value_struct,
            &target_lit,
            &mut ctx,
        )
        .unwrap();

        assert!(matches!(
            plan,
            ExprPlan::NoConstraint
        ));
    }

    #[test]
    fn geospatial_vars_include_temperature() {
        let meta = make_geo_meta();
        let dims: Vec<IStr> =
            vec!["lat".istr(), "lon".istr()];
        let mut ctx =
            LazyCompileCtx::new(&meta, &dims);

        let coord_struct =
            polars::prelude::as_struct(vec![
                col("lat"),
                col("lon"),
            ]);
        let value_struct =
            polars::prelude::as_struct(vec![
                col("temperature"),
            ]);
        let target_lit = make_target_struct_lit(
            &[45.0],
            &[170.0],
        );

        let plan = interpolate_selection_geospatial_lazy(
            &coord_struct,
            &value_struct,
            &target_lit,
            &mut ctx,
        )
        .unwrap();

        match plan {
            ExprPlan::Active { vars, .. } => {
                assert_vars_contain(
                    &vars,
                    "temperature",
                );
            }
            other => panic!(
                "expected Active plan, got {:?}",
                other
            ),
        }
    }
}
