use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;
use zarrs_object_store::object_store::ObjectStore;

mod adapters;
mod open_async;
mod open_sync;

pub use open_async::{open_store_async, open_store_from_object_store_async, AsyncOpenedStore};
pub use open_sync::{open_store, open_store_from_object_store, OpenedStore};

/// Input type for store parameters that can be either a URL string or an ObjectStore instance.
pub enum StoreInput {
    /// A URL string (e.g., "s3://bucket/path.zarr" or "/local/path.zarr")
    Url(String),
    /// An ObjectStore instance with an optional prefix path
    ObjectStore {
        store: Arc<dyn ObjectStore>,
        prefix: Option<String>,
    },
}

impl StoreInput {
    /// Extract from Python: accepts str or any ObjectStore (from rainbear or external libs like obstore).
    ///
    /// `AnyObjectStore` handles both:
    /// - `PyObjectStore` - stores created from rainbear's registered builders (full pooling)
    /// - `PyExternalObjectStore` - stores from external libs like obstore (recreated, no shared pool)
    pub fn from_py(store: &Bound<'_, PyAny>, prefix: Option<String>) -> PyResult<Self> {
        if let Ok(s) = store.extract::<String>() {
            Ok(StoreInput::Url(s))
        } else {
            let any_store: AnyObjectStore = store.extract()?;
            Ok(StoreInput::ObjectStore {
                store: any_store.into_dyn(),
                prefix,
            })
        }
    }

    /// Open as an async store.
    pub fn open_async(self) -> Result<AsyncOpenedStore, String> {
        match self {
            StoreInput::Url(url) => open_store_async(&url),
            StoreInput::ObjectStore { store, prefix } => {
                Ok(open_store_from_object_store_async(store, prefix))
            }
        }
    }

    /// Open as a sync store.
    pub fn open_sync(self) -> Result<OpenedStore, String> {
        match self {
            StoreInput::Url(url) => open_store(&url),
            StoreInput::ObjectStore { store, prefix } => open_store_from_object_store(store, prefix),
        }
    }
}

