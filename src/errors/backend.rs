use std::fmt::Debug;

use polars::prelude::{
    BooleanFunction, Expr, LiteralValue, Operator,
};
use pyo3::PyErr;
use snafu::Backtrace;
use snafu::prelude::*;

use crate::{
    IStr,
    chunk_plan::{
        ChunkGridSignature, ResolutionError,
    },
};
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
        "unsupported any value: {msg}",
        msg = msg,
    ))]
    UnsupportedAnyValue { msg: String },

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
        "unknown data variable: '{}' not found. Available variables: {:?}",
        name,
        available_vars
    ))]
    UnknownDataVar { name: IStr, available_vars: Vec<IStr> },

    #[snafu(display(
        "unknown dimension: '{}' not found. Available dimensions: {:?}",
        name,
        available_dimensions
    ))]
    UnknownDimension { name: IStr, available_dimensions: Vec<IStr> },

    #[snafu(display(
        "unknown zarr array: '{}' not found. Available zarr arrays: {:?}",
        name,
        available_zarr_arrays
    ))]
    UnknownZarrArray { name: IStr, available_zarr_arrays: Vec<IStr> },



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
    
    #[snafu(
        display(
            "{source} for dimensions {dims:?} with shape {shape:?} at paths {paths:?}",
            source = source,
            dims = dims,
            shape = shape,
            paths = paths,
        ),
        visibility(pub(crate))
    )]
    IncompatibleDimensionality { 
        backtrace: Backtrace,
        source: zarrs::array::IncompatibleDimensionalityError,
        dims: Vec<IStr>,
        shape: Vec<u64>,
        paths: Vec<IStr>,
    },
    #[snafu(display(
        "max_chunks_to_read exceeded: {total_chunks} chunks needed, limit is {max_chunks}",
        total_chunks = total_chunks,
        max_chunks = max_chunks,
    ),
    visibility(pub(crate))
)]
    MaxChunksToReadExceeded { total_chunks: usize, max_chunks: usize },
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
    #[snafu(
        display("polars error: {}", source),
        visibility(pub(crate))
    )]
    PolarsError { 
        source: polars::error::PolarsError,
        backtrace: Backtrace,
    },

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
    pub fn compile_polars(
        msg: String,
    ) -> BackendError {
        BackendError::CompileError { msg }
    }
    pub fn coord_not_found(
        msg: String,
    ) -> BackendError {
        BackendError::CoordNotFound { msg }
    }
    pub fn array_open_failed(
        msg: String,
    ) -> BackendError {
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
            >(format!(
                "{}",
                snafu::Report::from_error(error)
            )),
        }
    }
}

pub type BackendResult<T> =
    Result<T, BackendError>;
