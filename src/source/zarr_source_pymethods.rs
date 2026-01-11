#[pymethods]
impl ZarrSource {
    #[new]
    #[pyo3(signature = (zarr_url, batch_size, n_rows, variables=None, max_chunks_to_read=None))]
    fn new(
        zarr_url: String,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        variables: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
    ) -> PyResult<Self> {
        Self::new_impl(zarr_url, batch_size, n_rows, variables, max_chunks_to_read)
    }

    fn schema(&self) -> PySchema {
        self.schema_impl()
    }

    fn try_set_predicate(&mut self, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_set_predicate_impl(predicate)
    }

    fn set_with_columns(&mut self, columns: Vec<String>) {
        self.set_with_columns_impl(columns)
    }

    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        self.next_impl()
    }
}
