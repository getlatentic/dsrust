//! A faithful Rust reproduction of the [`json_repair`](https://pypi.org/project/json-repair/)
//! Python package, pinned to **0.61.7** — the malformed-JSON reader `dspy.JSONAdapter.parse` opens
//! with on every reply.
//!
//! It is not dspy's code and there is nothing upstream to copy: `json-repair` is a separate
//! dependency, and the Rust crates sharing its name are different implementations. Matching
//! "repairs malformed JSON" is not matching *these* repairs — the same reason `dsrust-tpe`
//! reproduces optuna's sampler rather than depending on a Bayesian-optimisation crate.
//!
//! ```
//! use json_repair::{Value, loads};
//!
//! let fields = loads(r#"{answer: "Paris", "why": 'it is the capital',}"#).expect("repaired");
//! assert_eq!(fields.get("answer"), Some(&Value::Str("Paris".into())));
//! ```
//!
//! # What is reproduced, and what is a seam
//!
//! The whole parser: the entry points, the repair heuristics, CPython's own `json` grammar for the
//! fast paths it takes, and the schema-guided repairs. The one thing that is *not* reproduced is
//! JSON Schema **validation** — upstream imports `jsonschema`, a separate package again, and
//! raises when it is absent. So this exposes [`SchemaValidator`] and, with none plugged in,
//! answers exactly as a Python environment without `jsonschema` does.
//!
//! # Two places Rust cannot follow Python
//!
//! - **A lone surrogate.** `"\ud800"` is a valid Python `str` and not a valid Rust `String`, so an
//!   escape naming one produces [`LONE_SURROGATE`] here. Python keeps the surrogate.
//! - **Recursion.** Python answers deep nesting with `RecursionError`, at a depth that belongs to
//!   the *caller's* stack rather than to the library: measured against 0.61.7 it is 330 nested
//!   arrays from a top-level call, 247 nested objects, and 317 arrays from forty frames deeper.
//!   There is no single number to match, and a Rust stack overflow is not catchable, so the limit
//!   here is the fixed [`MAX_DEPTH`].

mod dump;
mod parser;
mod pychar;
mod pynum;
mod schema;
mod strict_json;
mod value;

use std::cell::RefCell;
use std::rc::Rc;

pub use parser::LogEntry;
pub use schema::{SchemaRepairMode, SchemaValidator, ValidationError};
pub use value::{Object, Value};

/// The delimiters a string may open or close with, smart quotes among them.
pub(crate) const STRING_DELIMITERS: [char; 4] = ['"', '\'', '“', '”'];

/// What a `\u` escape naming an unpaired surrogate becomes, since a Rust `char` cannot hold one.
pub const LONE_SURROGATE: char = '\u{fffd}';

/// How deep a value may nest before the parse is refused. Chosen under the shallowest threshold
/// measured against Python — nested objects, at 247 — so this refuses where Python may also
/// refuse, rather than accepting where Python raises.
pub const MAX_DEPTH: usize = 240;

/// A JSON Schema node: an object, or one of the two boolean schemas.
pub(crate) type Schema = Value;

/// The repair log, shared between the parser and the schema repairer as upstream shares one list.
pub(crate) type LogSink = Option<Rc<RefCell<Vec<LogEntry>>>>;

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

pub type Result<T> = std::result::Result<T, Error>;

/// `json_repair.loads(text)` with its default arguments — the call `dspy.JSONAdapter.parse` makes.
pub fn loads(text: &str) -> Result<Value> {
    Repair::new().loads(text)
}

/// `json_repair.repair_json(text)`: the repaired value, written back out by `json.dumps`.
pub fn repair_json(text: &str) -> Result<String> {
    Repair::new().repair_json(text)
}

/// The keyword arguments `repair_json` takes, and the two entry points that read them.
#[derive(Default)]
pub struct Repair {
    skip_json_loads: bool,
    pub(crate) stream_stable: bool,
    pub(crate) strict: bool,
    schema: Option<Value>,
    mode: SchemaRepairMode,
    validator: Option<Rc<dyn SchemaValidator>>,
}

impl Repair {
    pub fn new() -> Self {
        Self::default()
    }

    /// Skips the whole-input `json.loads` check. The *suffix* fast path still applies: once the
    /// parser is past a prefix, decoding a valid value raw is still safe.
    pub fn skip_json_loads(mut self, skip: bool) -> Self {
        self.skip_json_loads = skip;
        self
    }

    /// For text that is a prefix of a longer reply still arriving, where trailing whitespace and a
    /// trailing backslash must survive rather than be tidied away.
    pub fn stream_stable(mut self, stable: bool) -> Self {
        self.stream_stable = stable;
        self
    }

    /// Surfaces structural problems — duplicate keys, missing separators, empty keys — instead of
    /// repairing them.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn schema_repair_mode(mut self, mode: SchemaRepairMode) -> Self {
        self.mode = mode;
        self
    }

    /// Plugs in the JSON Schema check upstream imports `jsonschema` for.
    pub fn validator(mut self, validator: Rc<dyn SchemaValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// The parsed value.
    pub fn loads(&self, text: &str) -> Result<Value> {
        self.run(text, None).map(|(value, _)| value)
    }

    /// The parsed value and the log of what was repaired to get it.
    pub fn loads_logged(&self, text: &str) -> Result<(Value, Vec<LogEntry>)> {
        let sink = Rc::new(RefCell::new(Vec::new()));
        let (value, log) = self.run(text, Some(sink))?;
        Ok((value, log))
    }

    /// The parsed value written back out with `json.dumps`, which is what upstream returns when
    /// `return_objects` is false — except for the empty string, which stays itself.
    pub fn repair_json(&self, text: &str) -> Result<String> {
        let value = self.loads(text)?;
        if value.is_empty_string() {
            return Ok(String::new());
        }
        Ok(value.to_string())
    }

    fn run(&self, text: &str, sink: LogSink) -> Result<(Value, Vec<LogEntry>)> {
        if self.schema.is_none() && self.mode == SchemaRepairMode::Salvage {
            return Err(Error::new("schema_repair_mode='salvage' requires schema."));
        }
        if self.schema.is_some() && self.strict {
            return Err(Error::new("schema and strict cannot be used together."));
        }
        let repairer = self.repairer(sink.clone())?;

        if let Some(value) = self.try_whole_input(text, repairer.as_deref())? {
            let log = sink.map(|sink| sink.borrow().clone()).unwrap_or_default();
            return Ok((value, log));
        }

        let mut parser = parser::Parser::new(text, self, sink.clone());
        parser.try_valid_json_suffix = true;
        let value = match (&repairer, &self.schema) {
            (Some(repairer), Some(schema)) => {
                let parsed = parser.parse_with_schema(repairer.clone(), schema.clone())?;
                repairer.validate(&parsed, Some(schema))?;
                parsed
            }
            _ => parser.parse()?,
        };
        let log = sink.map(|sink| sink.borrow().clone()).unwrap_or_default();
        Ok((value, log))
    }

    fn repairer(&self, sink: LogSink) -> Result<Option<Rc<schema::SchemaRepairer>>> {
        let Some(schema) = &self.schema else {
            return Ok(None);
        };
        let schema = schema_from_input(schema)?;
        let mut repairer = schema::SchemaRepairer::new(schema, sink, self.mode);
        if let Some(validator) = &self.validator {
            repairer = repairer.with_validator(validator.clone());
        }
        Ok(Some(Rc::new(repairer)))
    }

    /// The `json.loads` fast path, and the schema pass over what it returns.
    ///
    /// Upstream wraps all of this in `except (JSONDecodeError, TypeError, ValueError)`, so a
    /// refusal here is not a refusal of the input — it falls through to the parser. A repairer
    /// with no validator raises a `ValueError` and takes that route, and raises for real later,
    /// where the final `validate` is not guarded. What that `except` does *not* cover is a
    /// validator that could not read the schema, which reaches the caller from here.
    fn try_whole_input(
        &self,
        text: &str,
        repairer: Option<&schema::SchemaRepairer>,
    ) -> Result<Option<Value>> {
        if self.skip_json_loads {
            return Ok(None);
        }
        let Ok(parsed) = strict_json::loads(&text.chars().collect::<Vec<_>>()) else {
            return Ok(None);
        };
        let (Some(repairer), Some(schema)) = (repairer, &self.schema) else {
            return Ok(Some(parsed));
        };
        if caught(repairer.is_valid(&parsed, Some(schema)))?.unwrap_or(false) {
            return Ok(Some(parsed));
        }
        // A value the schema rejects is repaired and re-checked, and only a value that passes the
        // second check skips the parser.
        let Some(repaired) = caught(repairer.repair_value(Some(parsed), Some(schema), "$"))? else {
            return Ok(None);
        };
        let valid = caught(repairer.is_valid(&repaired, Some(schema)))?.unwrap_or(false);
        Ok(valid.then_some(repaired))
    }
}

/// `except (JSONDecodeError, TypeError, ValueError): pass`, and the inner `except ValueError`
/// around the repair attempt — the same set either way. Anything else propagates.
fn caught<T>(outcome: Result<T>) -> Result<Option<T>> {
    match outcome {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_foreign() => Err(error),
        Err(_) => Ok(None),
    }
}

/// `schema_from_input`: a JSON Schema dict or a boolean schema. Upstream also accepts a pydantic
/// model, which it converts by calling `model_json_schema()` — a caller here does that themselves.
fn schema_from_input(schema: &Value) -> Result<Value> {
    match schema {
        Value::Object(_) | Value::Bool(_) => Ok(schema.clone()),
        _ => Err(Error::new(
            "Schema must be a JSON Schema dict, boolean schema, or pydantic v2 model.",
        )),
    }
}
