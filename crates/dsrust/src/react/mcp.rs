//! MCP tools as [`Tool`](super::Tool)s — dspy's `convert_mcp_tool`, adapted to this crate's
//! client-agnostic shape.
//!
//! dspy couples its MCP support to the Python `mcp` library: `convert_mcp_tool` takes a live
//! `ClientSession` and calls it. This crate keeps the transport out — the same way
//! [`ChatModel`](crate::lm::ChatModel) keeps the provider out — and offers the two conversions dspy
//! performs around it: an MCP tool's input schema into the [`Tool::args`](super::Tool::args) map
//! ([`mcp_tool_args`], byte-verified against dspy), and an MCP call result into the observation a tool
//! returns ([`mcp_tool_result`]). A caller supplies the transport — any Rust MCP client, bridged to a
//! sync closure — and [`mcp_tool`] assembles the three into a [`Tool`](super::Tool) an agent can call.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use super::tool::FnTool;

/// dspy's `convert_input_schema_to_tool_args`: an MCP tool's input JSON schema as the `args` map a
/// [`Tool`](super::Tool) declares — each property under its name, every `$ref` resolved inline
/// against the schema's `$defs`. `required` does not appear here; which arguments are optional rides
/// on each property, matching dspy's `Tool.args`.
///
/// ```
/// use dsrust::mcp_tool_args;
///
/// // A `$ref` is resolved inline, because a model reading the args map cannot follow one.
/// let schema = serde_json::json!({
///     "properties": { "where": { "$ref": "#/$defs/Place" } },
///     "$defs": { "Place": { "type": "string", "description": "A city." } },
///     "required": ["where"],
/// });
/// let args = mcp_tool_args(&schema);
/// assert_eq!(args["where"]["type"], "string");
/// // `required` does not appear: which arguments are optional rides on each property, as
/// // `Tool.args` does upstream.
/// assert!(args.get("required").is_none());
/// ```
pub fn mcp_tool_args(input_schema: &Value) -> Value {
    let Some(properties) = input_schema.get("properties").and_then(Value::as_object) else {
        return json!({});
    };
    let defs = input_schema
        .get("$defs")
        .filter(|defs| defs.as_object().is_some_and(|defs| !defs.is_empty()));
    let args = properties.iter().map(|(name, property)| {
        let resolved = match defs {
            Some(defs) => resolve_refs(property, defs),
            None => property.clone(),
        };
        (name.clone(), resolved)
    });
    Value::Object(args.collect())
}

/// dspy's `_resolve_json_schema_reference`: replace every `{"$ref": ".../Name"}` with the definition
/// it names — keyed on the last path segment — recursively, so a reference inside an array or a
/// nested object is expanded too.
fn resolve_refs(value: &Value, defs: &Value) -> Value {
    match value {
        Value::Object(map) => {
            if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
                let name = reference.rsplit('/').next().unwrap_or_default();
                return resolve_refs(&defs[name], defs);
            }
            Value::Object(
                map.iter()
                    .map(|(key, value)| (key.clone(), resolve_refs(value, defs)))
                    .collect(),
            )
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(|item| resolve_refs(item, defs)).collect())
        }
        other => other.clone(),
    }
}

/// dspy's `_convert_mcp_tool_result`: an MCP `CallToolResult` — `{content: [...], isError}` on the
/// wire — as the observation a tool hands back. A lone text block is its bare string; several are
/// their list; content with no text is returned as it stands. An error result is an `Err` carrying
/// the same message dspy raises.
///
/// ```
/// use dsrust::mcp_tool_result;
///
/// // One text block is its bare string, not a one-element list — a model reads the observation,
/// // and a list of one would be noise.
/// let one = serde_json::json!({ "content": [{ "type": "text", "text": "Paris" }] });
/// assert_eq!(mcp_tool_result(&one).unwrap(), "Paris");
///
/// // An error result is an `Err` carrying what dspy raises, rather than an observation the model
/// // would read as an answer.
/// let failed = serde_json::json!({
///     "content": [{ "type": "text", "text": "no such place" }], "isError": true,
/// });
/// assert!(mcp_tool_result(&failed).is_err());
/// ```
pub fn mcp_tool_result(result: &Value) -> Result<String> {
    let content = result["content"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let texts: Vec<&str> = content
        .iter()
        .filter(|item| item["type"] == "text")
        .filter_map(|item| item["text"].as_str())
        .collect();
    let observation = match texts.as_slice() {
        [only] => (*only).to_owned(),
        [] => {
            let non_text: Vec<&Value> = content
                .iter()
                .filter(|item| item["type"] != "text")
                .collect();
            serde_json::to_string(&non_text).unwrap_or_default()
        }
        many => serde_json::to_string(many).unwrap_or_default(),
    };
    match result["isError"].as_bool() {
        Some(true) => Err(anyhow!("Failed to call a MCP tool: {observation}")),
        _ => Ok(observation),
    }
}

/// An MCP tool as a [`Tool`](super::Tool): its input schema becomes the argument contract, and each
/// call runs `transport` — the caller's bridge to an MCP client, taking the arguments and returning
/// the raw `CallToolResult` — with [`mcp_tool_result`] turning the result into the observation.
///
/// The transport is synchronous because [`Tool::call`](super::Tool::call) is; a caller driving an
/// async MCP client blocks on it (e.g. `Handle::current().block_on`, under a multi-threaded runtime).
///
/// `use<N, D, T>` on the return type, and it is load-bearing: an opaque return type captures every
/// lifetime in scope by default, so without it the tool borrows `input_schema` — which it never
/// holds, having copied the argument map out of it here. That made the one thing this function is
/// for impossible: fetch a schema, build a tool, hand the tool to an agent that outlives the
/// fetch. Every test in this crate built the tool and called it in one scope, so none of them
/// could see it. The two string parameters are named rather than `impl Into<String>` only because
/// a `use<...>` list has to name every type parameter in scope.
pub fn mcp_tool<N, D, T>(
    name: N,
    description: D,
    input_schema: &Value,
    transport: T,
) -> FnTool<impl Fn(&Value) -> Result<String> + Send + Sync + use<N, D, T>>
where
    N: Into<String>,
    D: Into<String>,
    T: Fn(&Value) -> Result<Value> + Send + Sync,
{
    let args = mcp_tool_args(input_schema);
    FnTool::new(name, description, args, move |args| {
        mcp_tool_result(&transport(args)?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::react::Tool;

    /// The tool outlives the schema it was built from, which is the only way it is ever used: a
    /// caller fetches a schema, builds a tool, and hands the tool to an agent that outlives the
    /// fetch.
    ///
    /// An opaque return type captures every lifetime in scope by default, so this did not compile
    /// until the return type said `use<N, D, T>`. Nothing here could see it — every other test
    /// builds the tool and calls it in one scope, where the borrow is still live.
    #[test]
    fn the_tool_outlives_the_schema_it_was_built_from() {
        fn built() -> Box<dyn Tool> {
            let schema = json!({ "properties": { "city": { "type": "string" } } });
            Box::new(mcp_tool("forecast", "Look it up.", &schema, |_| {
                Ok(json!("clear"))
            }))
        }
        let tool = built();
        assert_eq!(tool.name(), "forecast");
        assert_eq!(tool.args()["city"]["type"], "string");
    }

    /// Faithfulness to dspy's MCP support: our `mcp_tool_args` equals `convert_input_schema_to_tool_args`
    /// for the same input schema — properties under their names, `$ref`s resolved, `required` left off.
    /// The fixture is generated by running dspy (`scripts/generate_mcp_fixture.py`).
    #[test]
    fn our_args_match_dspy_convert_input_schema_to_tool_args() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/react/mcp_tool_args.json");
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
                .expect("fixture is valid json");
        for case in fixture["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("a case name");
            assert_eq!(
                mcp_tool_args(&case["input_schema"]),
                case["args"],
                "{name}: our mcp_tool_args diverges from dspy's"
            );
        }
    }

    #[test]
    fn a_lone_text_result_is_its_bare_string_and_an_error_is_an_err() {
        let ok = json!({ "content": [{ "type": "text", "text": "sunny, 22C" }], "isError": false });
        assert_eq!(mcp_tool_result(&ok).expect("ok"), "sunny, 22C");

        let failed =
            json!({ "content": [{ "type": "text", "text": "no such city" }], "isError": true });
        let error = mcp_tool_result(&failed).expect_err("an error result is an Err");
        assert!(
            error
                .to_string()
                .contains("Failed to call a MCP tool: no such city")
        );
    }

    /// An MCP tool reaches the agent as any other [`Tool`] does: the schema becomes its arguments, and
    /// a call runs the transport and reads its result back.
    #[test]
    fn an_mcp_tool_declares_its_arguments_and_calls_its_transport() {
        let schema = json!({ "type": "object", "properties": { "city": { "type": "string" } } });
        let tool = mcp_tool("get_weather", "look it up", &schema, |args| {
            let city = args["city"].as_str().unwrap_or("?");
            Ok(json!({ "content": [{ "type": "text", "text": format!("sunny in {city}") }] }))
        });
        assert_eq!(tool.name(), "get_weather");
        assert_eq!(tool.args(), &json!({ "city": { "type": "string" } }));
        assert_eq!(
            tool.call(&json!({ "city": "Paris" })).expect("calls"),
            "sunny in Paris"
        );
    }
}
