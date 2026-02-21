use pyo3::PyErr;
use snafu::prelude::*;
use polars::prelude::{Expr, LiteralValue, Operator, BooleanFunction};

use crate::{IStr, chunk_plan::{ChunkGridSignature, ResolutionError}};
/// Error type for backend operations.
#[derive(Debug, Snafu)]
pub enum BackendError {
    #[snafu(display(
        "unsupported polars expression: {expr:?} because {msg}",
        expr = expr,
        msg = msg,
    ))]
    UnsupportedPolarsExpression { msg: String, expr: Expr },

    #[snafu(display(
        "unsupported operator: {:?}",
        op
    ))]
    UnsupportedOperator { op: Operator },

    #[snafu(display(
        "unsupported literal: {lit:?}",
        lit = lit,
    ))]
    UnsupportedLiteral { lit: LiteralValue },

    #[snafu(display(
        "unsupported boolean function: {:?}",
        function
    ))]
    UnsupportedBooleanFunction { function: BooleanFunction },


    #[snafu(display("compile error: {}", msg))]
    CompileError { msg: String },
    #[snafu(display(
        "coordinate array not found: {}",
        msg
    ))]
    CoordNotFound { msg: String },
    #[snafu(display(
        "failed to open array: {}",
        msg
    ))]
    ArrayOpenFailed { msg: String },
    #[snafu(display(
        "failed to read chunk: {}",
        msg
    ))]
    ChunkReadFailed { msg: String },
    #[snafu(display("metadata not yet loaded"))]
    MetadataNotLoaded,


    #[snafu(display(
        "coord '{}' length mismatch: expected {}, got {}",
        name,
        expected_len,
        coord_len
    ))]
    CoordLengthMismatch { name: IStr, expected_len: u64, coord_len: u64 },

    #[snafu(display(
        "unknown variable: {}",
        name
    ))]
    UnknownVariable { name: IStr },

    #[snafu(display(
        "struct field path '{}' not found in metadata",
        path
    ))]
    StructFieldNotFound { path: IStr },

    #[snafu(display(
        "Missing chunk grid for signature: {}",
        sig
    ),
    visibility(pub(crate))
)]
    MissingChunkGrid { sig: ChunkGridSignature },


    #[snafu(display("other error: {}", msg))]
    Other { msg: String },
    
    #[snafu(context(false))]
    IncompatibleDimensionality { source: zarrs::array::IncompatibleDimensionalityError },
    #[snafu(context(false))]
    ArrayError { source: zarrs::array::ArrayError },
    #[snafu(context(false))]
    ArrayCreateError { source: zarrs::array::ArrayCreateError },
    #[snafu(context(false))]
    GroupCreateError { source: zarrs::group::GroupCreateError },
    #[snafu(context(false))]
    NodeCreateError { source: zarrs::hierarchy::NodeCreateError },

    #[snafu(context(false))]
    ParseError { source: url::ParseError },
    #[snafu(context(false))]
    PolarsError { source: polars::error::PolarsError },

    #[snafu(context(false))]
    ObjectStoreError { 
        #[snafu(source(from(zarrs_object_store::object_store::Error, Box::new)))]
        source: Box<zarrs_object_store::object_store::Error>
    },

    #[snafu(context(false))]
    FilesystemStoreCreateError { 
        #[snafu(source(from(zarrs::filesystem::FilesystemStoreCreateError, Box::new)))]
        source: Box<zarrs::filesystem::FilesystemStoreCreateError>
    },


    #[snafu(context(false))]
    ResolutionError { source: ResolutionError },

    #[snafu(display(
        "invalid regex pattern '{}': {}",
        pattern,
        source
    ),
    visibility(pub(crate))
)]
    RegexError { source: regex::Error, pattern: String },

    #[snafu(display(
        "chunk dimension too large: {} > {}",
        dim,
        max
    ),
    visibility(pub(crate))
)]
    ChunkDimTooLarge { 
        source: std::num::TryFromIntError,
        dim: u64, max: usize },

    #[snafu(display(
        "invalid file URL: {}",
        url
    ),
    visibility(pub(crate))
)]
    InvalidFileUrl { url: String },

    #[snafu(display(
        "failed to create tokio runtime for sync store: {}",
        source
    ),
    visibility(pub(crate))
)]
    CreateTokioRuntimeForSyncStore { source: std::io::Error,
    prefix: String,
    store: String,
    
     },
}


impl BackendError {
    pub fn other(msg: String) -> BackendError {
        BackendError::Other { msg }
    }
    pub fn compile_polars(msg: String) -> BackendError {
        BackendError::CompileError { msg }
    }
    pub fn coord_not_found(msg: String) -> BackendError {
        BackendError::CoordNotFound { msg }
    }
    pub fn array_open_failed(msg: String) -> BackendError {
        BackendError::ArrayOpenFailed { msg }
    }
}
/// To Py Error
impl From<BackendError> for PyErr {
    fn from(error: BackendError) -> PyErr {
        match error {
            BackendError::CoordNotFound {
                msg,
            } => PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(format!(
                "coordinate array not found: {}",
                msg
            )),
            _ => PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(error.to_string()),
        }
    }
}


pub type BackendResult<T> =
    Result<T, BackendError>;
