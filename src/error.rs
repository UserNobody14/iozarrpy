#[derive(Debug)]
pub(crate) enum Error {
    Message(String),
}

impl<E: std::fmt::Display> From<E> for Error {
    fn from(e: E) -> Self {
        Error::Message(e.to_string())
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

