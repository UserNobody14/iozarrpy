use std::fmt::Display;

use pyo3::PyErr;

/// Error type for backend operations.
#[derive(Debug, Clone)]
pub enum BackendError {
    // Unsupported polars expression.
    UnsupportedPolarsExpression(String),

    /// Compile error.
    CompileError(String),
    /// The requested coordinate array was not found.
    CoordNotFound(String),
    /// Failed to open the zarr array.
    ArrayOpenFailed(String),
    /// Failed to read chunk data.
    ChunkReadFailed(String),
    /// Metadata not yet loaded.
    MetadataNotLoaded,
    /// Other error.
    Other(String),
    // /// Incompatible dimensionality error.
    // ZarrError(ZarrErrors),
}

impl Display for BackendError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            BackendError::UnsupportedPolarsExpression(expr) => {
                write!(
                    f,
                    "unsupported polars expression: {}",
                    expr
                )
            }
            BackendError::CompileError(err) => {
                write!(
                    f,
                    "compile error: {}",
                    err
                )
            }
            BackendError::CoordNotFound(dim) => {
                write!(
                    f,
                    "coordinate array not found: {}",
                    dim
                )
            }
            BackendError::ArrayOpenFailed(
                msg,
            ) => {
                write!(
                    f,
                    "failed to open array: {}",
                    msg
                )
            }
            BackendError::ChunkReadFailed(
                msg,
            ) => {
                write!(
                    f,
                    "failed to read chunk: {}",
                    msg
                )
            }
            BackendError::MetadataNotLoaded => {
                write!(
                    f,
                    "metadata not yet loaded"
                )
            }
            BackendError::Other(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {}

/// To Py Error
impl From<BackendError> for PyErr {
    fn from(error: BackendError) -> PyErr {
        match error {
            BackendError::CoordNotFound(msg) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "coordinate array not found: {}",
                    msg
                ))
            }
            _ => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                error.to_string(),
            ),
        }
    }
}

// Auto-convert zarr errors to backend errors
impl From<zarrs::array::ArrayCreateError>
    for BackendError
{
    fn from(
        error: zarrs::array::ArrayCreateError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<zarrs::group::GroupCreateError>
    for BackendError
{
    fn from(
        error: zarrs::group::GroupCreateError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<zarrs::hierarchy::NodeCreateError>
    for BackendError
{
    fn from(
        error: zarrs::hierarchy::NodeCreateError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<zarrs::array::IncompatibleDimensionalityError>
    for BackendError
{
    fn from(
        error: zarrs::array::IncompatibleDimensionalityError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<zarrs::array::ArrayError>
    for BackendError
{
    fn from(
        error: zarrs::array::ArrayError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<zarrs::filesystem::FilesystemStoreCreateError>
    for BackendError
{
    fn from(
        error: zarrs::filesystem::FilesystemStoreCreateError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<zarrs_object_store::object_store::Error>
    for BackendError
{
    fn from(
        error: zarrs_object_store::object_store::Error,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

impl From<url::ParseError> for BackendError {
    fn from(
        error: url::ParseError,
    ) -> BackendError {
        BackendError::Other(error.to_string())
    }
}

pub type BackendResult<T> =
    Result<T, BackendError>;
