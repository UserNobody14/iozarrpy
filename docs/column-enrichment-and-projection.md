# Column enrichment, projection, and streaming batches

This document states the **behavioral contract** for how Rainbear chooses what to read, what to attach after chunk merge (enrichment), which streaming chunk groups are kept, and how final column order is produced.

Implementation lives primarily in:

- `src/scan/column_policy.rs` — resolved policy, shared 1D availability rules
- `src/shared/structural.rs` — `expand_io_source_physical`, struct/dim expansion, implied 1D aux
- `src/chunk_plan/mod.rs` — `predicate_column_refs`
- `src/scan/enrich_df_vars.rs` — post-merge 1D gather
- `src/backend/implementation/iterating_common.rs` — group filter, output columns, batch pipeline

---

## 1. Why this exists

- **Multiple chunk grids:** Variables may sit on different consolidated groups (e.g. main field on `time × lead_time × point`, aux `station_id` only on `point`). A batch from the main group may lack aux columns; those are **enriched** by reading the full 1D Zarr array and **gathering** along an index column already in the batch.
- **Polars IO pushdown:** `with_columns` may omit filter-only or CF-listed names; the **physical superset** still includes predicate columns, all dataset dimensions, and implied 1D aux on touched dims.
- **CF auxiliary coordinates** may be in `node.arrays` but not `data_vars`; enrichment iterates the expanded wishlist, not `data_vars` alone.

---

## 2. Core algebra

| Concept | Meaning |
|--------|--------|
| **`predicate_refs`** | `predicate_column_refs(expr)` — every column name referenced in the pushed-down filter. |
| **`physical_superset`** | `expand_io_source_physical(with_cols, &predicate_refs, meta)`: union of `with_cols`, predicate refs, **all** `meta.dim_analysis.all_dims`, struct-expanded flat paths, plus **1D arrays** whose sole dim appears in “dims used” by that set (`projection_dims_used` + `expand_1d_aux_on_projection_dims`). |
| **`ResolvedColumnPolicy`** | Holds `predicate_refs` + `physical_superset` for a given `Expr` and effective `with_columns` (see streaming/sync below). |
| **Group can satisfy `name`** | `group_supplies_array_or_1d_enrichable`: the group’s `vars` contains `name`, or `name` is **1D** and its dim is in the group’s **signature dims** (same rule as enrichment from a batch that already has that dim column). |

---

## 3. Sync full scan (`lazy.rs`)

- **`chunk_expanded`:** `with_columns.map(|c| expand_projection_to_flat_paths(c, meta))` — **narrow** per-chunk mask so heterogeneous grids are not over-constrained at read time.
- **Enrichment wishlist:** `ResolvedColumnPolicy::new(with_columns.clone().or_else(|| Some(tidy names as set)), &expr, &meta).physical_superset` — same **physical superset** as streaming uses after merge.

Final hierarchical restructure runs after enrichment. The Python sync entry may apply `filter(predicate)` after the Rust scan returns; enrichment still runs **before** that filter on the frame Rust returns when applicable.

---

## 4. Streaming iterator (`iterating.rs`, `icechunk_iterating.rs`)

- **`effective_with_columns`:** Polars list, or full `tidy_column_order(None)` if Polars passed `None`.
- **`expanded_with_columns`:** `ResolvedColumnPolicy::new(effective set as `BTreeSet`, &expr, &meta).physical_superset` — drives chunk read hints and enrichment.
- **`filter_streaming_grid_groups(groups, policy.predicate_refs(), raw_polars_with_columns, meta)`:** Predicate columns must be satisfiable per group (dims in signature; arrays via direct read or **1D + dim in signature**). **Output column loop** uses **raw** Polars `with_columns` only when `Some`; when `None`, no per-output-name rejection — required for heterogeneous grids (temperature vs pressure on different chunk signatures).

---

## 5. Output column order (`output_columns_for_streaming_batch`)

- **`None` from Polars:** start from `tidy_column_order`, then append extras from `physical_superset` not already listed.
- **`Some(req)`:** preserve `req` order; append **1D non-dimension** names from `physical_superset` not in `req` (e.g. CF lat/lon omitted from the IO callback).

`project_to_polars_output` drops missing names without error; empty projection list leaves the frame unchanged.

---

## 6. Batch pipeline (`combine_and_postprocess_batch`)

Fixed order:

1. Merge chunk DataFrames for the batch.
2. Enrich (e.g. `enrich_df_missing_requested_vars` / `_async` with `expanded_with_columns`).
3. `filter(state.predicate)`.
4. Row limit, optional `restructure_to_structs`, `project_to_polars_output`.

Enrichment **must** run before the predicate so filter columns exist.

---

## 7. Enrichment (`enrich_df_vars.rs`)

- **`with_columns: None`:** no-op.
- For each missing `var` in the set: use **`one_d_var_enrichable_from_columns`** (1D array whose dim column is present). **`dim_row_indices_for_enrichment`** returns `Ok(None)` if the dim column dtype cannot drive a gather (skip); **hard errors** (e.g. coord shape mismatch) propagate. **`append_gathered_1d_var`** performs read → `gather_by` → `apply_encoding` → `with_column`.

---

## 8. Checklist for changes

1. CF-only-in-`arrays` variables: still in **tidy_schema** / expansion from `all_zarr_array_paths`.
2. **1D on a dim:** candidate for enrichment and for `expand_1d_aux_on_projection_dims`.
3. **Heterogeneous streaming:** never feed **full tidy** list into `filter_streaming_grid_groups`’s output-column requirement; use raw Polars list or skip that loop when `None`.
4. **Sync vs streaming parity:** same **physical superset** for enrichment after merge.

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **Enrichment** | Post-merge: read full 1D Zarr, gather rows using an existing dim column. |
| **Physical superset** | Expanded names for reads + enrichment (not necessarily final Polars output). |
| **Effective with-columns** | Streaming: Polars list or full tidy order when Polars passes `None`. |
