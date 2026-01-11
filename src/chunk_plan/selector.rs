use super::compile_node::compile_node;
use super::errors::{CompileError, CoordIndexResolver};
use super::prelude::*;
use super::selection::DatasetSelection;

pub(super) fn compile_selector(
    selector: &Selector,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    dim_lengths: &[u64],
    vars: &[String],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<DatasetSelection, CompileError> {
    match selector {
        Selector::Union(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            Ok(left_node.union(&right_node))
        }
        Selector::Difference(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            Ok(left_node.difference(&right_node))
        }
        Selector::ExclusiveOr(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            Ok(left_node
                .difference(&right_node)
                .union(&right_node.difference(&left_node)))
        }
        Selector::Intersect(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            Ok(left_node.intersect(&right_node))
        }
        Selector::Empty => Ok(DatasetSelection::empty()),
        Selector::ByName { names, .. } => {
            let vars = names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            Ok(DatasetSelection::all_for_vars(vars))
        }
        Selector::ByIndex { .. }
        | Selector::Matches(_)
        | Selector::ByDType(_)
        | Selector::Wildcard => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::*;
    use crate::meta::ZarrDatasetMeta;

    struct DummyResolver;
    impl CoordIndexResolver for DummyResolver {
        fn index_range_for_value_range(
            &mut self,
            _dim: &str,
            _range: &super::super::types::ValueRange,
        ) -> Result<Option<super::super::types::IndexRange>, super::super::errors::ResolveError> {
            Ok(None)
        }
    }

    #[test]
    fn selector_by_name_limits_variable_set() {
        let meta = ZarrDatasetMeta {
            arrays: BTreeMap::new(),
            dims: vec![],
            data_vars: vec![],
        };
        let dims: Vec<String> = vec![];
        let dim_lengths: Vec<u64> = vec![];
        let vars: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut resolver = DummyResolver;

        let names: Arc<[polars::prelude::PlSmallStr]> = Arc::from(
            vec![
                polars::prelude::PlSmallStr::from("a"),
                polars::prelude::PlSmallStr::from("c"),
            ]
            .into_boxed_slice(),
        );
        let sel = Selector::ByName { names, strict: true };

        let out = compile_selector(&sel, &meta, &dims, &dim_lengths, &vars, &mut resolver).unwrap();
        assert!(out.0.contains_key("a"));
        assert!(out.0.contains_key("c"));
        assert!(!out.0.contains_key("b"));
    }
}
