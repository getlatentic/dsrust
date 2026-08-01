//! What a refusal is, and where each kind of one is caught.
//!
//! The original raises `ValueError` for an ordinary refusal, `SchemaDefinitionError` for a schema
//! it cannot read, and lets whatever the validator raises through untouched. Those are three
//! different things only because `except ValueError` appears at six places and each one changes
//! the answer — so the distinction is not cosmetic, and Rust has no exception hierarchy to carry
//! it implicitly.

/// A refusal, carrying the message upstream's exception carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
    kind: Kind,
}

/// Which of upstream's exceptions this is, which decides where it is *caught*.
///
/// `except ValueError` appears at six places in the library and each one changes the answer, so
/// the distinction is not cosmetic: a schema branch that fails is tried again, and a validator
/// that cannot answer at all is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A `ValueError`: the ordinary refusal, caught wherever upstream catches one.
    Value,
    /// `SchemaDefinitionError`, a `ValueError` subclass the array and list-mapping paths re-raise
    /// before they would catch it — but which the union branches do *not*, since they catch the
    /// base class alone.
    Definition,
    /// Something that is not a `ValueError` at all — `jsonschema` refusing a schema it cannot
    /// read, which nothing in `json_repair` catches and which reaches the caller.
    Foreign,
}

impl Error {
    pub(crate) fn new(message: &str) -> Self {
        Self {
            message: message.to_owned(),
            kind: Kind::Value,
        }
    }

    /// `SchemaDefinitionError`: the schema itself is wrong.
    pub(crate) fn definition(message: &str) -> Self {
        Self {
            message: message.to_owned(),
            kind: Kind::Definition,
        }
    }

    /// The validator could not answer, which is not a refusal of the value.
    pub(crate) fn foreign(message: &str) -> Self {
        Self {
            message: message.to_owned(),
            kind: Kind::Foreign,
        }
    }

    pub(crate) fn is_definition(&self) -> bool {
        self.kind == Kind::Definition
    }

    /// Whether `except ValueError` would let this through.
    pub(crate) fn is_foreign(&self) -> bool {
        self.kind == Kind::Foreign
    }

    /// The message the original's `ValueError` carries.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// What every entry point returns.
pub type Result<T> = std::result::Result<T, Error>;
