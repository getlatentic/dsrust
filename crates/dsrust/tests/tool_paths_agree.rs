//! The same tool, declared three ways, renders one roster line.
//!
//! A `Tool` can be built from a Rust function (`#[tool]`, read at compile time), from a struct of
//! arguments (`arguments_schema`, also compile time), or from a JSON schema that arrived at run
//! time from an MCP server (`mcp_tool_args`). They are genuinely different mechanisms — a macro
//! cannot read a schema a server has not sent yet — but they end in the same place: the roster
//! text a model reads, `It takes arguments {…}`.
//!
//! Two of the three had divergences from dspy, found separately and months apart in effort: the
//! macro path ordered its keys as a signature's field note rather than as pydantic writes them,
//! and the struct path titled every parameter because it rendered them as *properties* of a model
//! instead of as roots. Nothing compared the three against each other, so a fix to one could
//! silently leave the others behind.

use dsrust::react::tool_args;
use dsrust::signature::arguments_schema;
use dsrust::{Tool, mcp_tool_args, tool};
use serde_json::json;

/// Look up the weather.
#[tool]
fn forecast(city: String, days: i64) -> anyhow::Result<String> {
    let _ = (city, days);
    Ok("clear".to_owned())
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct ForecastArgs {
    city: String,
    days: i64,
}

/// What the roster prints for a tool's arguments.
fn roster(args: &serde_json::Value) -> String {
    serde_json::to_string(args).expect("serializes")
}

/// All three, byte for byte.
///
/// Compared as text: under `preserve_order` two objects differing only in key order are equal, and
/// the key order is what the roster prints.
#[test]
fn a_function_a_struct_and_a_schema_agree() {
    let from_macro = roster(Forecast.args());
    let from_struct = roster(&serde_json::Value::Object(
        arguments_schema::<ForecastArgs>().expect("a struct has fields"),
    ));
    let from_schema = roster(&mcp_tool_args(&json!({
        "properties": {
            "city": { "type": "string" },
            "days": { "type": "integer" },
        },
        "required": ["city", "days"],
    })));

    assert_eq!(from_macro, from_struct, "the macro and the struct disagree");
    assert_eq!(
        from_macro, from_schema,
        "the macro and an MCP schema disagree"
    );
    assert_eq!(
        from_macro,
        r#"{"city":{"type":"string"},"days":{"type":"integer"}}"#
    );
}

/// `tool_args` is the fourth spelling of the struct path and must not have drifted from it.
#[test]
fn the_two_struct_spellings_agree() {
    assert_eq!(
        roster(&tool_args::<ForecastArgs>()),
        roster(&serde_json::Value::Object(
            arguments_schema::<ForecastArgs>().expect("a struct has fields")
        ))
    );
}
