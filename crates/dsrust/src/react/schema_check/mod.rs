//! python-jsonschema's `validate`, as far as the message it raises.
//!
//! dspy checks each tool argument with `jsonschema.validate(instance, schema)`, which collects every
//! error the draft 2020-12 keywords yield — in the order the schema states its keywords — and
//! raises `best_match` of them. The message of that one error is what the model reads. This module
//! reproduces the errors, their paths and their `context`, and the `best_match` heuristic that
//! picks among them.
//!
//! Not reproduced: `unevaluatedItems`, `unevaluatedProperties` and `$dynamicRef`, which no schema a
//! parameter type produces carries; and `format`, which `validate` does not assert by default.

mod keywords;
#[cfg(test)]
mod tests;
mod values;

use serde_json::Value;

/// One `ValidationError`: its message, its path relative to the instance it was raised on, the
/// keyword that raised it, the errors it holds as context, and whether the instance is of the type
/// its schema states.
#[derive(Debug, Clone)]
pub(crate) struct Error {
    pub(crate) message: String,
    path: Vec<Step>,
    validator: &'static str,
    context: Vec<Error>,
    matches_type: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Step {
    Index(usize),
    Key(String),
}

/// `best_match(iter_errors(instance)).message`, or `None` when the instance is valid.
pub(crate) fn message(instance: &Value, schema: &Value) -> Option<String> {
    let walk = Walk { root: schema };
    best_match(walk.errors(instance, schema)).map(|error| error.message)
}

/// The validator's state: the root schema, which `$ref` resolves against.
pub(super) struct Walk<'a> {
    root: &'a Value,
}

impl Walk<'_> {
    /// `iter_errors`: each keyword of the schema, in the schema's own order.
    pub(super) fn errors(&self, instance: &Value, schema: &Value) -> Vec<Error> {
        match schema {
            Value::Bool(true) => Vec::new(),
            Value::Bool(false) => vec![Error::new(
                format!(
                    "False schema does not allow {}",
                    crate::python::repr(instance)
                ),
                "false",
                instance,
                schema,
            )],
            Value::Object(keywords) => keywords
                .iter()
                .flat_map(|(keyword, stated)| {
                    keywords::apply(self, keyword, stated, instance, schema)
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// `descend`: the errors of a subschema, their paths extended by the step taken to reach the
    /// child instance when one was taken.
    pub(super) fn descend(
        &self,
        instance: &Value,
        schema: &Value,
        step: Option<Step>,
    ) -> Vec<Error> {
        let mut errors = self.errors(instance, schema);
        if let Some(step) = step {
            for error in &mut errors {
                error.path.insert(0, step.clone());
            }
        }
        errors
    }

    pub(super) fn is_valid(&self, instance: &Value, schema: &Value) -> bool {
        self.errors(instance, schema).is_empty()
    }

    /// `_validate_reference` for a reference into this document: a JSON pointer from the root.
    pub(super) fn reference(&self, reference: &str, instance: &Value) -> Vec<Error> {
        let Some(pointer) = reference.strip_prefix('#') else {
            return Vec::new();
        };
        let unescaped = pointer.replace("~1", "/").replace("~0", "~");
        match self.root.pointer(&unescaped) {
            Some(schema) => self.errors(instance, schema),
            None => Vec::new(),
        }
    }
}

impl Error {
    pub(super) fn new(
        message: String,
        validator: &'static str,
        instance: &Value,
        schema: &Value,
    ) -> Self {
        Error {
            message,
            path: Vec::new(),
            validator,
            context: Vec::new(),
            matches_type: matches_type(instance, schema),
        }
    }

    pub(super) fn with_context(mut self, context: Vec<Error>) -> Self {
        self.context = context;
        self
    }

    /// `by_relevance()`: shallower first, then earlier, then a keyword that is not `anyOf` or
    /// `oneOf`, then one whose schema type the instance is not.
    fn relevance(&self) -> (i64, Vec<Step>, bool, bool, bool) {
        let weak = matches!(self.validator, "anyOf" | "oneOf");
        (
            -(self.path.len() as i64),
            self.path.clone(),
            !weak,
            false,
            !self.matches_type,
        )
    }
}

/// `ValidationError._matches_type`: the instance is of a type the error's own schema states.
fn matches_type(instance: &Value, schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => values::is_type(instance, kind),
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter_map(Value::as_str)
            .any(|kind| values::is_type(instance, kind)),
        _ => false,
    }
}

/// `best_match`: the most relevant error, then down through its context while one child is more
/// relevant than the next.
fn best_match(errors: Vec<Error>) -> Option<Error> {
    let mut best = errors.into_iter().reduce(|best, candidate| {
        match candidate.relevance() > best.relevance() {
            true => candidate,
            false => best,
        }
    })?;
    while !best.context.is_empty() {
        let mut smallest: Vec<Error> = best.context.clone();
        smallest.sort_by_key(Error::relevance);
        if smallest.len() >= 2 && smallest[0].relevance() == smallest[1].relevance() {
            return Some(best);
        }
        best = smallest.swap_remove(0);
    }
    Some(best)
}
