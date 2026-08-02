//! Read JSON that is not quite JSON.
//!
//! A port of the Python package [`json-repair`](https://pypi.org/project/json-repair/), pinned to
//! **0.61.7** and held to its answers byte for byte. Give it a trailing comma, a missing brace, a
//! bare key, single quotes, smart quotes, a code fence, or prose wrapped around the whole thing,
//! and it returns the value the text was reaching for.
//!
//! ```
//! use json_repair::{Value, loads};
//!
//! let value = loads(r#"{answer: "Paris", "why": 'the capital',}"#)?;
//! assert_eq!(value.get("answer"), Some(&Value::Str("Paris".into())));
//! # Ok::<(), json_repair::Error>(())
//! ```
//!
//! Language models are the usual source of such text, but nothing here knows that: the input is a
//! string and the output is a value.
//!
//! # Why a port rather than another repairing parser
//!
//! Several Rust crates repair malformed JSON, and each repairs it its own way. This one is not a
//! new set of heuristics — it is *these* heuristics, so a program moving from Python gets the same
//! answers for the same input, including the strange ones. Where the original is surprising, so is
//! this. That is the whole promise, and [the test suite](https://github.com/getlatentic/dsrust)
//! checks it by running the Python package and comparing the bytes.
//!
//! # Getting a value out
//!
//! [`loads`] returns a [`Value`], the Rust spelling of what Python's `json` module produces —
//! object, array, string, integer, float, boolean, null. [`repair_json`] returns the text
//! `json.dumps` would write for it instead, which is useful when the destination is another
//! program that wants JSON.
//!
//! ```
//! assert_eq!(json_repair::repair_json("{a: 1,}")?, r#"{"a": 1}"#);
//! # Ok::<(), json_repair::Error>(())
//! ```
//!
//! [`Repair`] carries the options and is the entry point for anything past the defaults:
//!
//! | | |
//! |---|---|
//! | [`loads`] / [`Repair::loads`] | text in, a [`Value`] out |
//! | [`repair_json`] / [`Repair::repair_json`] | text in, `json.dumps` text out |
//! | [`from_file`] / [`Repair::from_reader`] | a file or a reader, which parses *slightly* differently |
//! | [`Repair::loads_logged`] | the value, and every repair that produced it |
//! | [`Repair::loads_as`] | straight into your own type, with the `serde` feature |
//! | [`Repair::dumps`] | the writer on its own, for a value from elsewhere |
//!
//! The options are [`Repair::strict`], [`Repair::stream_stable`], [`Repair::skip_json_loads`],
//! [`Repair::ensure_ascii`], and [`Repair::schema`] with [`Repair::validator`].
//!
//! ```
//! use json_repair::Repair;
//!
//! let reader = Repair::new().strict(false).ensure_ascii(false);
//! assert_eq!(reader.repair_json("{答案: '北京',}")?, r#"{"答案": "北京"}"#);
//!
//! let (value, repairs) = reader.loads_logged("{a: 1")?;
//! assert_eq!(value.get("a"), Some(&json_repair::Value::Int(1)));
//! assert!(repairs.iter().any(|entry| entry.text.contains("literal instead of a quote")));
//! # Ok::<(), json_repair::Error>(())
//! ```
//!
//! With the `serde` feature on, [`Value`] implements `Serialize` and `Deserialize`, converts to and
//! from `serde_json::Value`, and [`Repair::loads_as`] deserializes straight into your own type.
//!
//! # What is reproduced, and what is a seam
//!
//! The whole parser: both entry points, every repair heuristic, CPython's own `json` grammar for
//! the fast paths the parser takes, and the schema-guided repairs.
//!
//! The one thing not reproduced is JSON Schema **validation**. The Python package delegates that
//! to `jsonschema`, a separate library, and raises when it is not installed. Reimplementing a
//! validator here would be a different project, so it is a [`SchemaValidator`] you plug in — and
//! with none plugged in this answers exactly as a Python environment without `jsonschema` does.
//!
//! # Two places Rust cannot follow Python
//!
//! - **A lone surrogate.** `"\ud800"` is a valid Python `str` and not a valid Rust `String`, so an
//!   escape naming one yields [`LONE_SURROGATE`] here where Python keeps the surrogate.
//! - **Recursion depth.** Python answers deep nesting with `RecursionError`, at a depth belonging
//!   to the *caller's* stack rather than to the library — measured against 0.61.7 at 330 nested
//!   arrays from a top-level call, 247 nested objects, and 317 arrays from forty frames deeper.
//!   There is no single number to match, and a Rust stack overflow cannot be caught, so the limit
//!   here is the fixed [`MAX_DEPTH`].
#![deny(missing_docs)]

mod dump;
mod error;
mod parser;
pub mod pychar;
mod pynum;
mod repair;
mod schema;
#[cfg(feature = "serde")]
mod serde_support;
mod strict_json;
mod value;

pub use error::{Error, Result};
pub use parser::LogEntry;
pub use repair::{Repair, from_file, loads, repair_json};
pub use schema::{SchemaRepairMode, SchemaValidator, ValidationError};
pub use value::{Object, Value};

#[cfg(feature = "serde")]
pub use serde_support::loads_as;

/// A JSON Schema node: an object, or one of the two boolean schemas.
pub(crate) type Schema = Value;

/// The repair log, shared between the parser and the schema repairer as the original shares one
/// list between them.
pub(crate) type LogSink = Option<std::rc::Rc<std::cell::RefCell<Vec<LogEntry>>>>;

/// The delimiters a string may open or close with, smart quotes among them.
pub(crate) const STRING_DELIMITERS: [char; 4] = ['"', '\'', '“', '”'];

/// What a `\u` escape naming an unpaired surrogate becomes, since a Rust `char` cannot hold one.
pub const LONE_SURROGATE: char = '\u{fffd}';

/// How deep a value may nest before the parse is refused. Chosen under the shallowest threshold
/// measured against Python — nested objects, at 247 — so this refuses where Python may also
/// refuse, rather than accepting where Python raises.
pub const MAX_DEPTH: usize = 240;
