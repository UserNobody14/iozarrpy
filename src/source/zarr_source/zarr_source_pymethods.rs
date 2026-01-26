use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_polars::{PyDataFrame, PySchema};

use crate::store::StoreInput;

use super::ZarrSource;

#[pymethods]
impl ZarrSource {
    #[new]
    #[pyo3(signature = (store, batch_size, n_rows, variables=None, max_chunks_to_read=None, prefix=None))]
    fn new(
        store: &Bound<'_, PyAny>,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        variables: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
        prefix: Option<String>,
    ) -> PyResult<Self> {
        let store_input = StoreInput::from_py(store, prefix)?;
        Self::new_impl(store_input, batch_size, n_rows, variables, max_chunks_to_read)
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

