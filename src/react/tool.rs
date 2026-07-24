//! The tool contract: what the agent may call, and how each tool reaches the model.
//!
//! dspy takes any Python callable and inspects it for a name and an argument schema. Here a
//! tool is a trait, so its name, description and argument schema are declared rather than
//! derived, and the compiler checks the implementation.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::signature::json_field_schema;

/// Something the agent can call. dspy derives these from a callable's signature; declaring
/// them keeps the argument contract visible to both the model and the compiler.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;

    /// Shown to the model when it chooses. A tool nobody can tell apart from another will not
    /// be chosen correctly, so this earns its place in the prompt.
    fn description(&self) -> &str;

    /// dspy's `Tool.args`: a JSON object mapping each argument name to that argument's JSON
    /// Schema. It is rendered into the instructions, so a model that has never seen this tool
    /// still knows what to send. A tool that takes nothing returns an empty object.
    ///
    /// Required rather than defaulted: a tool whose arguments go undeclared is one the model
    /// can only guess at, which is the failure this trait exists to prevent.
    fn args(&self) -> &Value;

    /// Run with the arguments the model supplied, returning the observation it will read.
    fn call(&self, args: &Value) -> Result<String>;

    /// dspy's `Tool.__call__` returns whatever the tool produced, which need not be text: an
    /// observation may be any JSON, and ReActV2's `submit` returns the final-output mapping the
    /// loop reads back as the answer. `call` is the common case — a string observation — so it
    /// stays the required method and this defaults to wrapping what it returns; a tool whose result
    /// is structured overrides this instead.
    fn call_value(&self, args: &Value) -> Result<Value> {
        self.call(args).map(Value::String)
    }
}

/// The `args` map for a tool whose arguments are the fields of `T`, so the schema comes from a
/// Rust type instead of a hand-written literal that can drift from the code reading it. dspy
/// reads the same map off a Python function's type hints.
///
/// Carries per-argument schemas only, matching dspy's `Tool.args`: which arguments are
/// optional shows up as a `default` on the argument, never as a separate `required` list.
pub fn tool_args<T: schemars::JsonSchema>() -> Value {
    json_field_schema::<T>()
        .get("properties")
        .cloned()
        .unwrap_or_else(|| json!({}))
}

/// The name the model uses to say it is done. dspy adds this tool itself, so the model always
/// has a way to stop that is indistinguishable from any other choice it makes.
pub const FINISH: &str = "finish";

/// One line of the tool catalogue, matching dspy's `Tool.__str__`: the name, the description
/// in `<desc>` tags, and the argument schema the model has to fill.
pub(super) fn describe(name: &str, description: &str, args: &Value) -> String {
    let desc = match description.is_empty() {
        true => ".".to_owned(),
        // dspy flattens newlines so a multi-line description cannot break the numbered list.
        false => format!(", whose description is <desc>{description}</desc>.").replace('\n', "  "),
    };
    format!("{name}{desc} It takes arguments {}.", python_repr(args))
}

/// Render a JSON value the way Python's `repr` prints a dict, because that is literally what
/// dspy interpolates into the instructions — `str(tool)` formats `self.args`, a dict.
///
/// The difference is visible to the model: `{'city': {'type': 'string'}}` rather than
/// `{"city":{"type":"string"}}`. Matching it keeps the prompt bytes identical, which is the
/// standard the conformance fixtures hold everything else to.
fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!("'{key}': {}", python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        number => number.to_string(),
    }
}

/// A tool built from a closure, for callers who do not want to declare a type per tool.
pub struct FnTool<F> {
    pub name: String,
    pub description: String,
    /// dspy's `Tool.args`: argument name to that argument's JSON Schema. Build it from a type
    /// with [`tool_args`], or write the object out for a one-argument tool.
    pub args: Value,
    pub call: F,
}

impl<F> FnTool<F>
where
    F: Fn(&Value) -> Result<String> + Send + Sync,
{
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        args: Value,
        call: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            args,
            call,
        }
    }
}

impl<F> Tool for FnTool<F>
where
    F: Fn(&Value) -> Result<String> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        (self.call)(args)
    }
}

/// Read a required string argument, with an error the model can act on.
pub fn arg_str<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string argument `{name}`"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_description_spanning_lines_stays_on_one_catalogue_line() {
        // dspy replaces newlines so a wrapped docstring cannot break the numbered list apart.
        let entry = describe("noisy", "first\nsecond", &json!({}));
        assert_eq!(
            entry,
            "noisy, whose description is <desc>first  second</desc>. It takes arguments {}."
        );
    }

    #[test]
    fn a_tool_with_no_description_is_rendered_without_the_desc_tags() {
        assert_eq!(
            describe("bare", "", &json!({})),
            "bare. It takes arguments {}."
        );
    }

    #[test]
    fn tool_args_reads_the_argument_schema_off_a_rust_type() {
        // The point of the helper: the schema cannot drift from the struct the tool parses.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct WeatherArgs {
            city: String,
            days: u8,
        }
        assert_eq!(
            tool_args::<WeatherArgs>(),
            json!({
                "city": { "type": "string" },
                "days": { "type": "integer", "format": "uint8", "minimum": 0, "maximum": 255 },
            })
        );
    }
}
