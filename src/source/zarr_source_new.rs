impl ZarrSource {
    fn new_impl(
        zarr_url: String,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        variables: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
    ) -> PyResult<Self> {
        let (opened, meta) = crate::meta::open_and_load_dataset_meta(&zarr_url)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let store = opened.store.clone();

        let vars = if let Some(v) = variables {
            v
        } else {
            meta.data_vars.clone()
        };

        if vars.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "no variables found/selected",
            ));
        }

        // Primary var defines the chunk iteration.
        let primary_path = meta
            .arrays
            .get(&vars[0])
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable"))?
            .path
            .clone();
        let primary = Array::open(store.clone(), &primary_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        let primary_grid_shape = primary.chunk_grid().grid_shape().to_vec();

        // Polars may pass 0 to mean "unspecified"; interpret it as the default.
        let n_rows_left = match n_rows {
            None | Some(0) => usize::MAX,
            Some(n) => n,
        };
        let batch_size = match batch_size {
            None | Some(0) => DEFAULT_BATCH_SIZE,
            Some(n) => n,
        };

        let dims = meta
            .arrays
            .get(&vars[0])
            .map(|m| m.dims.clone())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| (0..primary.dimensionality()).map(|i| format!("dim_{i}")).collect());
        let chunk_iter = ChunkPlan::all(primary_grid_shape.clone()).into_index_iter();

        Ok(Self {
            meta,
            store,
            dims,
            vars,
            batch_size,
            n_rows_left,
            predicate: None,
            with_columns: None,
            primary_grid_shape,
            chunk_iter,
            current_chunk_indices: None,
            chunk_offset: 0,
            done: primary.dimensionality() == 0 && false,
            chunks_left: max_chunks_to_read,
        })
    }

    fn consume_chunk_budget(&mut self) -> PyResult<()> {
        if let Some(left) = self.chunks_left {
            if left == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_chunks_to_read limit hit",
                ));
            }
            self.chunks_left = Some(left - 1);
        }
        Ok(())
    }

    fn schema_impl(&self) -> PySchema {
        let schema = self.meta.tidy_schema(Some(&self.vars));
        PySchema(Arc::new(schema))
    }

}
