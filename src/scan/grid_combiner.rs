//! Utilities for combining DataFrames from multiple chunk grids.
//!
//! When variables have the same dimensions but different chunk shapes, they are
//! read from separate grids. This module provides utilities to join the results
//! on dimension columns.

use crate::IStr;
use polars::prelude::*;

/// Combines DataFrames from multiple grids by joining on dimension columns.
///
/// # Arguments
/// * `dfs` - Vector of DataFrames, each containing dimension columns + that grid's variables
/// * `dim_columns` - Names of dimension columns to join on
///
/// # Returns
/// A single DataFrame with all variables joined on dimension columns
pub fn join_grid_dataframes(
    dfs: Vec<DataFrame>,
    dim_columns: &[IStr],
) -> PolarsResult<DataFrame> {
    if dfs.is_empty() {
        return Ok(DataFrame::empty());
    }
    if dfs.len() == 1 {
        return Ok(dfs
            .into_iter()
            .next()
            .unwrap());
    }

    let dim_col_strs: Vec<&str> = dim_columns
        .iter()
        .map(|s| s.as_ref())
        .collect();

    let mut iter = dfs.into_iter();
    let mut result = iter.next().unwrap();

    // Join each subsequent DataFrame on dimension columns
    for df in iter {
        // Get the column names that are NOT dimension columns (these are the variable columns)
        let right_var_cols: Vec<&str> = df
            .get_column_names()
            .into_iter()
            .map(|name| name.as_str())
            .filter(|name| {
                !dim_col_strs.contains(name)
            })
            .collect();

        if right_var_cols.is_empty() {
            // No new columns to add, skip
            continue;
        }

        // Perform outer join on dimension columns
        result = result.join(
            &df,
            &dim_col_strs,
            &dim_col_strs,
            JoinArgs::new(JoinType::Full)
                .with_coalesce(
                    JoinCoalesce::CoalesceColumns,
                ),
            None,
        )?;
    }

    Ok(result)
}

/// Combines DataFrames by joining on dimension columns - LazyFrame version.
///
/// More efficient for larger datasets as it allows Polars to optimize the join.
pub fn join_grid_lazyframes(
    lfs: Vec<LazyFrame>,
    dim_columns: &[IStr],
) -> PolarsResult<LazyFrame> {
    if lfs.is_empty() {
        return Ok(DataFrame::empty().lazy());
    }
    if lfs.len() == 1 {
        return Ok(lfs
            .into_iter()
            .next()
            .unwrap());
    }

    let dim_exprs: Vec<Expr> = dim_columns
        .iter()
        .map(|s| {
            let name: &str = s.as_ref();
            col(name)
        })
        .collect();

    let mut iter = lfs.into_iter();
    let mut result = iter.next().unwrap();

    // Join each subsequent LazyFrame on dimension columns
    for lf in iter {
        result = result.join(
            lf,
            dim_exprs.clone(),
            dim_exprs.clone(),
            JoinArgs::new(JoinType::Full)
                .with_coalesce(
                    JoinCoalesce::CoalesceColumns,
                ),
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoIStr;
    use polars::df;

    #[test]
    fn test_join_single_df() {
        let df = df! {
            "y" => [0i64, 0, 1, 1],
            "x" => [0i64, 1, 0, 1],
            "temp" => [1.0f64, 2.0, 3.0, 4.0],
        }
        .unwrap();

        let dims = vec!["y".istr(), "x".istr()];
        let result = join_grid_dataframes(
            vec![df.clone()],
            &dims,
        )
        .unwrap();

        assert_eq!(result.height(), 4);
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_join_two_dfs_same_coords() {
        let df1 = df! {
            "y" => [0i64, 0, 1, 1],
            "x" => [0i64, 1, 0, 1],
            "temp" => [1.0f64, 2.0, 3.0, 4.0],
        }
        .unwrap();

        let df2 = df! {
            "y" => [0i64, 0, 1, 1],
            "x" => [0i64, 1, 0, 1],
            "pressure" => [100.0f64, 200.0, 300.0, 400.0],
        }
        .unwrap();

        let dims = vec!["y".istr(), "x".istr()];
        let result = join_grid_dataframes(
            vec![df1, df2],
            &dims,
        )
        .unwrap();

        assert_eq!(result.height(), 4);
        assert_eq!(result.width(), 4); // y, x, temp, pressure
    }
}
