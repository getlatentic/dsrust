//! A tool built from its argument type, with no macro of this crate's in the way.
//!
//! dspy reads a tool's name, description and argument schema off a callable — `__name__`,
//! `__doc__`, the type hints. Rust erases a doc comment before the program runs, so none of that
//! survives on a `fn`. It does survive on a *type*: `#[derive(JsonSchema)]` records the doc comment
//! as the schema's `description`, the fields as its `properties`, each field's own doc comment as
//! that property's `description`, and the type's name as its `title`. All four are read back here.
//!
//! So this is the whole of what `#[tool]` does, out of a derive the ecosystem already has. What it
//! asks in exchange is a struct per tool; what it gives back is prose per argument, which dspy
//! keeps in `Tool.arg_desc` and a Rust function cannot carry — parameters take no doc comment.

use anyhow::Result;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use super::tool::{FnTool, Tool};
use super::tool_call::parsed_args;

/// A tool whose name, description and argument schema are all its argument type's.
///
/// ```
/// use dsrust::{Tool, typed_tool};
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// /// Append one instructional block to the draft and return its id.
/// #[derive(Deserialize, JsonSchema)]
/// struct AddBlock {
///     /// One of: explanation, worked_example.
///     block_type: String,
///     text: String,
/// }
///
/// let tool = typed_tool(|args: AddBlock| Ok(format!("Added {}.", args.block_type)));
/// assert_eq!(tool.name(), "add_block");
/// assert_eq!(
///     tool.description(),
///     "Append one instructional block to the draft and return its id."
/// );
/// assert_eq!(tool.args()["block_type"]["description"], "One of: explanation, worked_example.");
/// ```
pub fn typed_tool<T, F>(call: F) -> impl Tool
where
    T: DeserializeOwned + JsonSchema,
    F: Fn(T) -> Result<String> + Send + Sync,
{
    let schema = described_schema::<T>();
    let name = snake(text_at(&schema, "title"));
    let declared = schema
        .get("properties")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let required: Vec<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let called = name.clone();
    let checked = declared.clone();
    FnTool::new(
        name,
        text_at(&schema, "description"),
        declared,
        // What dspy's `Tool.__call__` raises for a wrong or missing argument, raised here too: the
        // loop records it as `Execution error in …`, which is what the model reads and retries from.
        move |args: &Value| {
            let names: Vec<&str> = required.iter().map(String::as_str).collect();
            let parsed = parsed_args(&called, args, &checked, &names)?;
            call(serde_json::from_value::<T>(Value::Object(parsed))?)
        },
    )
}

/// The type's own schema, keeping the root metadata `json_field_schema` drops: a field slot has no
/// use for the title and description, and a tool is made of them.
fn described_schema<T: JsonSchema>() -> Map<String, Value> {
    let generator = schemars::generate::SchemaSettings::default()
        .with(|settings| {
            settings.inline_subschemas = true;
            settings.meta_schema = None;
        })
        .into_generator();
    match generator.into_root_schema_for::<T>().to_value() {
        Value::Object(object) => object,
        _ => Map::new(),
    }
}

fn text_at(schema: &Map<String, Value>, key: &str) -> String {
    schema
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// `AddBlock` becomes `add_block`, the way dspy takes a tool's name from the callable's own
/// `__name__` rather than asking for it a second time.
fn snake(name: String) -> String {
    let characters: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(characters.len() + 4);
    for (index, character) in characters.iter().enumerate() {
        if character.is_uppercase() && index > 0 && breaks_word(&characters, index) {
            out.push('_');
        }
        out.extend(character.to_lowercase());
    }
    out
}

/// An uppercase letter starts a word when the one before it is lowercase, or when it is the last
/// of a run of capitals and the next is lowercase — so `HttpGet` and `HTTPGet` both split once.
fn breaks_word(characters: &[char], index: usize) -> bool {
    let after_lower = characters[index - 1].is_lowercase();
    let ends_a_run = characters
        .get(index + 1)
        .is_some_and(|next| next.is_lowercase());
    after_lower || ends_a_run
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four things dspy reads off a callable, read off a type instead.
    #[derive(serde::Deserialize, JsonSchema)]
    struct SetTitle {
        /// The learner-facing title.
        title: String,
        heading: Option<String>,
    }

    #[test]
    fn a_type_carries_everything_a_tool_needs() {
        let tool = typed_tool(|args: SetTitle| Ok(format!("Title set to {:?}.", args.title)));
        assert_eq!(tool.name(), "set_title");
        assert_eq!(tool.args()["title"]["type"], "string");
        assert_eq!(
            tool.args()["title"]["description"],
            "The learner-facing title."
        );
        assert_eq!(
            tool.call(&json!({ "title": "Fractions" }))
                .expect("answers"),
            "Title set to \"Fractions\"."
        );
    }

    /// An optional argument may be left out; a required one that is missing, or one of the wrong
    /// type, raises what dspy raises.
    #[test]
    fn a_bad_argument_raises_what_dspy_raises() {
        let tool = typed_tool(|args: SetTitle| Ok(args.heading.unwrap_or(args.title)));
        assert_eq!(
            tool.call(&json!({ "title": "Fractions" }))
                .expect("answers"),
            "Fractions"
        );
        let missing = tool.call(&json!({})).expect_err("raises");
        assert_eq!(
            format!("{missing:#}"),
            "TypeError: set_title() missing 1 required positional argument: 'title'"
        );
        let wrong = tool.call(&json!({ "title": 7 })).expect_err("raises");
        assert_eq!(
            format!("{wrong:#}"),
            "ValueError: Arg title is invalid: 7 is not of type 'string'"
        );
    }

    /// The closure captures, which is how a roster of these comes to share one state.
    #[test]
    fn a_typed_tool_captures_its_state() {
        let written = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let held = std::sync::Arc::clone(&written);
        let tool = typed_tool(move |args: SetTitle| {
            held.lock().expect("not poisoned").push(args.title.clone());
            Ok(args.title)
        });
        tool.call(&json!({ "title": "first" })).expect("answers");
        assert_eq!(*written.lock().expect("not poisoned"), ["first"]);
    }

    #[test]
    fn a_run_of_capitals_splits_once() {
        assert_eq!(snake("AddBlock".to_owned()), "add_block");
        assert_eq!(snake("HTTPGet".to_owned()), "http_get");
        assert_eq!(snake("Read".to_owned()), "read");
    }
}
