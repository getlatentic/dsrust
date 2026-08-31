//! The tool contract: what the agent may call, and how each tool reaches the model.
//!
//! dspy takes any Python callable and inspects it for a name and an argument schema. Here a
//! tool is a trait, so its name, description and argument schema are declared rather than
//! derived, and the compiler checks the implementation.

use std::future::Future;
use std::pin::Pin;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

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

    /// dspy's `Tool.acall`: the same call, awaited.
    ///
    /// A tool that reaches a network, a database or a subprocess is a future, and upstream awaits
    /// one whenever the wrapped callable returns a coroutine. The default answers with whatever
    /// [`call_value`](Tool::call_value) returns, so a tool with nothing to await writes nothing —
    /// which is upstream's own "allow calling a sync tool in the async path".
    ///
    /// Boxed rather than an `async fn`, because the agents hold `dyn Tool`.
    fn acall_value<'a>(
        &'a self,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move { self.call_value(args) })
    }
}

/// The `args` map for a tool whose arguments are the fields of `T`, so the schema comes from a
/// Rust type instead of a hand-written literal that can drift from the code reading it. dspy
/// reads the same map off a Python function's type hints.
///
/// Carries per-argument schemas only, matching dspy's `Tool.args`: which arguments are
/// optional shows up as a `default` on the argument, never as a separate `required` list.
pub fn tool_args<T: schemars::JsonSchema>() -> Value {
    let Some(properties) = crate::signature::arguments_schema::<T>() else {
        return json!({});
    };
    Value::Object(properties)
}

/// The name the model uses to say it is done. dspy adds this tool itself, so the model always
/// has a way to stop that is indistinguishable from any other choice it makes.
pub const FINISH: &str = "finish";

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

/// A tool whose body is a future — one that reaches a network, a database or a subprocess.
///
/// dspy's equivalent is a tool whose callable is `async def`: `acall` awaits it, and `__call__`
/// hands back the un-awaited coroutine. Here the synchronous half refuses in words instead, since
/// every agent awaits [`Tool::acall_value`](Tool::acall_value) and nothing else should be calling
/// this one.
pub struct AsyncFnTool<F> {
    pub name: String,
    pub description: String,
    pub args: Value,
    pub call: F,
}

impl<F, Answering> AsyncFnTool<F>
where
    F: Fn(Value) -> Answering + Send + Sync,
    Answering: Future<Output = Result<String>> + Send,
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

impl<F, Answering> Tool for AsyncFnTool<F>
where
    F: Fn(Value) -> Answering + Send + Sync,
    Answering: Future<Output = Result<String>> + Send,
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

    fn call(&self, _args: &Value) -> Result<String> {
        Err(anyhow!(
            "`{}` is asynchronous, so it answers through `acall_value` — which is what every agent \
             calls",
            self.name
        ))
    }

    fn acall_value<'a>(
        &'a self,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move { (self.call)(args.clone()).await.map(Value::String) })
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

    #[test]
    fn tool_args_reads_the_argument_schema_off_a_rust_type() {
        // The point of the helper: the schema cannot drift from the struct the tool parses.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct WeatherArgs {
            city: String,
            days: u8,
        }
        // No `format`. dspy builds `Tool.args` from the callable's annotations and writes none —
        // measured, `dspy.Tool(get_weather).args` is `{"city": {"type": "string"}, "days":
        // {"type": "integer"}}` — and this map is prompt text, rendered as "It takes arguments".
        // This assertion carried `"format": "uint8"` until that was checked against upstream.
        //
        // The bounds stay: `u8` really is bounded where Python's `int` is not, which is a
        // difference between the two type systems rather than between the two dialects.
        // Compared as text, not as a `Value`. Under `preserve_order` two objects differing only
        // in key order are equal, and the key order is what dspy prints — so a `Value` assertion
        // here reads as strict and checks everything except the thing that ends up in the prompt.
        assert_eq!(
            serde_json::to_string(&tool_args::<WeatherArgs>()).expect("serializes"),
            r#"{"city":{"type":"string"},"days":{"maximum":255,"minimum":0,"type":"integer"}}"#
        );
    }
}
