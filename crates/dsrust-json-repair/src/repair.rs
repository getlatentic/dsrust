//! The options every entry point reads, and the walk they configure.
//!
//! One builder rather than the seven keyword arguments `repair_json` takes, because most callers
//! want none of them and the rest want one.

use std::cell::RefCell;
use std::rc::Rc;

use crate::parser::LogEntry;
use crate::schema::SchemaRepairMode;
use crate::value::Value;
use crate::{Error, LogSink, Result, SchemaValidator, parser, schema, strict_json};

/// Whether a value that is already valid JSON, found after a prefix, is decoded by CPython's
/// scanner or read by the repair parser. Upstream ties this to where the input came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Suffix {
    /// `loads`: decode it.
    Decode,
    /// `load` and `from_file`: repair it.
    Repair,
}

/// Read a value out of text that may not be valid JSON.
///
/// `json_repair.loads(text)` with its default arguments. Text with nothing readable in it yields
/// the empty string rather than an error, which is what the original does.
///
/// ```
/// use json_repair::{Value, loads};
///
/// assert_eq!(loads("[1, 2,")?, Value::Array(vec![Value::Int(1), Value::Int(2)]));
/// assert_eq!(loads("no json here")?, Value::Str(String::new()));
/// # Ok::<(), json_repair::Error>(())
/// ```
pub fn loads(text: &str) -> Result<Value> {
    Repair::new().loads(text)
}

/// Read a value out of text that may not be valid JSON, and write it back out as JSON.
///
/// `json_repair.repair_json(text)`. The bytes are Python's `json.dumps`: `", "` between items,
/// `": "` after a key, every code point outside `\x20`-`\x7e` escaped.
///
/// ```
/// assert_eq!(json_repair::repair_json("{'a': [1, 2,}")?, r#"{"a": [1, 2]}"#);
/// # Ok::<(), json_repair::Error>(())
/// ```
pub fn repair_json(text: &str) -> Result<String> {
    Repair::new().repair_json(text)
}

/// Read a value out of a file whose contents may not be valid JSON.
///
/// `json_repair.from_file(filename)`. See [`Repair::from_file`] for why this is not the same as
/// reading the file and calling [`loads`].
pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Value> {
    Repair::new().from_file(path)
}

/// The keyword arguments `repair_json` takes, and the two entry points that read them.
#[derive(Default)]
pub struct Repair {
    skip_json_loads: bool,
    ensure_ascii: bool,
    pub(crate) stream_stable: bool,
    pub(crate) strict: bool,
    schema: Option<Value>,
    mode: SchemaRepairMode,
    validator: Option<Rc<dyn SchemaValidator>>,
}

impl Repair {
    /// The default arguments: no schema, not strict, not streaming.
    pub fn new() -> Self {
        Self {
            ensure_ascii: true,
            ..Self::default()
        }
    }

    /// `json.dumps`'s `ensure_ascii`, which [`Repair::repair_json`] forwards. On — the default, as
    /// it is Python's — every code point outside `\x20`-`\x7e` leaves as an escape.
    ///
    /// ```
    /// use json_repair::Repair;
    ///
    /// assert_eq!(Repair::new().repair_json("{'k': '统一码'}")?, r#"{"k": "\u7edf\u4e00\u7801"}"#);
    /// assert_eq!(
    ///     Repair::new().ensure_ascii(false).repair_json("{'k': '统一码'}")?,
    ///     r#"{"k": "统一码"}"#,
    /// );
    /// # Ok::<(), json_repair::Error>(())
    /// ```
    ///
    /// The other `**json_dumps_args` upstream forwards are `json.dumps`'s surface rather than this
    /// library's, and are not carried: dump the [`Value`] yourself if you need them.
    pub fn ensure_ascii(mut self, ascii: bool) -> Self {
        self.ensure_ascii = ascii;
        self
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

    /// Repairs the value against a JSON Schema: coercing scalars, filling defaults, dropping what
    /// the schema forbids. Needs a [`Repair::validator`] to check the result against.
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// How hard the schema pass tries before giving up on a value.
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
        self.run(text, None, Suffix::Decode).map(|(value, _)| value)
    }

    /// The parsed value and the log of what was repaired to get it.
    pub fn loads_logged(&self, text: &str) -> Result<(Value, Vec<LogEntry>)> {
        let sink = Rc::new(RefCell::new(Vec::new()));
        let (value, log) = self.run(text, Some(sink), Suffix::Decode)?;
        Ok((value, log))
    }

    /// The value in a file, as `json_repair.from_file(filename)` reads it.
    ///
    /// Not the same as reading the file and calling [`Repair::loads`]. Upstream turns the suffix
    /// fast path *off* for file input — `try_valid_json_suffix = json_fd is None` — so a valid
    /// JSON value after a prefix goes through the repair parser rather than CPython's scanner, and
    /// the two disagree on about one input in five hundred.
    pub fn from_file(&self, path: impl AsRef<std::path::Path>) -> Result<Value> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|error| Error::new(&format!("{}: {error}", path.as_ref().display())))?;
        self.from_text_of_a_file(&text)
    }

    /// The value in a reader, as `json_repair.load(fd)` reads it.
    ///
    /// The whole reader is read before parsing. Upstream reads it in chunks through a wrapper that
    /// implements only indexing and length, so the parse sees the same characters either way and
    /// the difference is memory rather than behaviour — which is also why `chunk_length` is not
    /// carried.
    pub fn from_reader<R: std::io::Read>(&self, mut reader: R) -> Result<Value> {
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .map_err(|error| Error::new(&error.to_string()))?;
        self.from_text_of_a_file(&text)
    }

    /// A value written out as `json.dumps` writes it, honouring [`Repair::ensure_ascii`].
    ///
    /// The writer on its own, for a value that came from somewhere else. [`Repair::repair_json`]
    /// is this applied to what [`Repair::loads`] read.
    ///
    /// ```
    /// use json_repair::{Repair, Value};
    ///
    /// // `1e+16`, not Rust's `10000000000000000`.
    /// assert_eq!(Repair::new().dumps(&Value::Float(1e16)), "1e+16");
    /// ```
    pub fn dumps(&self, value: &Value) -> String {
        crate::dump::dumps(value, self.ensure_ascii)
    }

    fn from_text_of_a_file(&self, text: &str) -> Result<Value> {
        self.run(text, None, Suffix::Repair).map(|(value, _)| value)
    }

    /// The parsed value written back out with `json.dumps`, which is what upstream returns when
    /// `return_objects` is false — except for the empty string, which stays itself.
    pub fn repair_json(&self, text: &str) -> Result<String> {
        let value = self.loads(text)?;
        if value.is_empty_string() {
            return Ok(String::new());
        }
        Ok(crate::dump::dumps(&value, self.ensure_ascii))
    }

    fn run(&self, text: &str, sink: LogSink, suffix: Suffix) -> Result<(Value, Vec<LogEntry>)> {
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
        parser.try_valid_json_suffix = suffix == Suffix::Decode;
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
        Err(error) if !error.is_caught_by_repair_json() => Err(error),
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
