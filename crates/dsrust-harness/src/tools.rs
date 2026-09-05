//! dsrust tools offered to a coding agent, in this process.
//!
//! The agent sees one MCP server; each call comes back here and runs the
//! dsrust [`Tool`] it names. Nothing is spawned and nothing listens on a port —
//! agent-harness serves the server over the agent's own protocol.

use dsrust::Tool;
use harness::{HostTool, ToolServer};
use serde_json::{Value, json};

/// `tools` as one host tool server the agent sees as `name` — the same
/// `Vec<Box<dyn Tool>>` a `ReActV2!` takes, so one roster serves either loop.
///
/// Call this inside a tokio runtime when any tool is async: the agent invokes
/// tools from a thread of its own, and the runtime the caller is on is what
/// their futures need. A sync tool — the common case — needs nothing.
///
/// Keep that runtime running while the agent works — `await` the run, as
/// [`HarnessModel`](crate::HarnessModel) does. Blocking the runtime's thread
/// until the run finishes (a `join`, a `block_on` inside `block_on`) stops the
/// timers and sockets an async tool is waiting on while the run waits on the
/// tool: a deadlock, and a quiet one.
pub fn tool_server(
    name: impl Into<String>,
    tools: impl IntoIterator<Item = Box<dyn Tool>>,
) -> ToolServer {
    server(name, tools, false)
}

/// As [`tool_server`], with every tool declared read-only.
///
/// dsrust tools carry no such declaration, so [`tool_server`] treats them as
/// mutating — which the built-in OpenAI-compatible agent withholds from a
/// read-only run. Say so here for tools that only look things up, and they are
/// offered there too.
pub fn read_only_tool_server(
    name: impl Into<String>,
    tools: impl IntoIterator<Item = Box<dyn Tool>>,
) -> ToolServer {
    server(name, tools, true)
}

fn server(
    name: impl Into<String>,
    tools: impl IntoIterator<Item = Box<dyn Tool>>,
    read_only: bool,
) -> ToolServer {
    let runtime = tokio::runtime::Handle::try_current().ok();
    tools
        .into_iter()
        .fold(ToolServer::new(name), |server, tool| {
            server.with_tool(Hosted {
                tool,
                runtime: runtime.clone(),
                read_only,
            })
        })
}

/// One dsrust tool behind the host-tool contract.
struct Hosted {
    tool: Box<dyn Tool>,
    runtime: Option<tokio::runtime::Handle>,
    read_only: bool,
}

impl HostTool for Hosted {
    fn name(&self) -> &str {
        self.tool.name()
    }
    fn description(&self) -> &str {
        self.tool.description()
    }
    fn input_schema(&self) -> Value {
        input_schema(self.tool.args())
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
    fn call(&self, arguments: Value) -> Result<String, String> {
        // The agent calls from a thread that is not a runtime worker, so
        // blocking here blocks nothing else: a tokio handle drives the future
        // on the caller's runtime; without one, pollster drives a future that
        // has no reactor to wait on — which is every sync tool's default.
        let call = self.tool.acall_value(&arguments);
        let observed = match &self.runtime {
            Some(handle) => handle.block_on(call),
            None => pollster::block_on(call),
        };
        match observed {
            Ok(Value::String(text)) => Ok(text),
            Ok(value) => Ok(value.to_string()),
            Err(error) => Err(error.to_string()),
        }
    }
}

/// dsrust's `Tool::args` — a name → JSON Schema map, dspy's `Tool.args` — as an
/// MCP `inputSchema`.
///
/// dspy drops `required` and lets optionality ride on each property, so it has
/// to be read back off them: an argument whose schema names a `default` is
/// optional, and every other one is required. That is the only signal the map
/// carries; an argument that is optional in the tool's own mind but declares no
/// default is required here, and the agent will supply one.
pub fn input_schema(args: &Value) -> Value {
    let properties = args.as_object().cloned().unwrap_or_default();
    let required: Vec<&String> = properties
        .iter()
        .filter(|(_, schema)| schema.get("default").is_none())
        .map(|(name, _)| name)
        .collect();
    json!({ "type": "object", "properties": properties, "required": required })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsrust::FnTool;

    fn lookup() -> Box<dyn Tool> {
        Box::new(FnTool::new(
            "lookup",
            "look something up",
            json!({ "query": { "type": "string" }, "limit": { "type": "integer", "default": 5 } }),
            |args: &Value| Ok(format!("found {}", args["query"].as_str().unwrap_or("?"))),
        ))
    }

    #[test]
    fn required_is_rebuilt_from_the_arguments_that_declare_no_default() {
        let schema = input_schema(lookup().args());
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        assert_eq!(
            schema["required"],
            json!(["query"]),
            "limit has a default, so it is optional"
        );
        assert_eq!(
            input_schema(&json!({}))["required"],
            json!([]),
            "a tool that takes nothing requires nothing"
        );
    }

    #[test]
    fn a_hosted_tool_keeps_its_name_and_description_and_runs_the_dsrust_call() {
        let server = tool_server("shop", [lookup()]);
        assert_eq!(server.name(), "shop");
        let tool = &server.tools()[0];
        assert_eq!(
            (tool.name(), tool.description()),
            ("lookup", "look something up")
        );
        assert!(!tool.read_only());
        assert_eq!(
            tool.input_schema()["required"],
            json!(["query"]),
            "the schema the agent sees"
        );
        assert_eq!(
            tool.call(json!({ "query": "cats" })),
            Ok("found cats".to_owned())
        );
        assert!(read_only_tool_server("shop", [lookup()]).tools()[0].read_only());
    }

    #[test]
    fn a_structured_observation_is_serialised_and_an_error_is_the_agents_to_read() {
        struct Structured;
        impl Tool for Structured {
            fn name(&self) -> &str {
                "structured"
            }
            fn description(&self) -> &str {
                "answers with an object, or refuses"
            }
            fn args(&self) -> &Value {
                static ARGS: Value = Value::Null;
                &ARGS
            }
            fn call(&self, _: &Value) -> anyhow::Result<String> {
                unreachable!("call_value is overridden")
            }
            fn call_value(&self, args: &Value) -> anyhow::Result<Value> {
                match args["fail"].as_bool() {
                    Some(true) => Err(anyhow::anyhow!("closed")),
                    _ => Ok(json!({ "units": 3 })),
                }
            }
        }
        let server = tool_server("s", [Box::new(Structured) as Box<dyn Tool>]);
        let tool = &server.tools()[0];
        assert_eq!(tool.call(json!({})), Ok(r#"{"units":3}"#.to_owned()));
        assert_eq!(tool.call(json!({ "fail": true })), Err("closed".to_owned()));
    }

    #[tokio::test]
    async fn an_async_tool_runs_on_the_callers_runtime_from_the_agents_thread() {
        struct Sleepy;
        impl Tool for Sleepy {
            fn name(&self) -> &str {
                "sleepy"
            }
            fn description(&self) -> &str {
                "awaits the runtime"
            }
            fn args(&self) -> &Value {
                static ARGS: Value = Value::Null;
                &ARGS
            }
            fn call(&self, _: &Value) -> anyhow::Result<String> {
                unreachable!("the async path is the one under test")
            }
            fn acall_value<'a>(
                &'a self,
                _: &'a Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = anyhow::Result<Value>> + Send + 'a>,
            > {
                Box::pin(async {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    Ok(json!("woke"))
                })
            }
        }
        // Built inside the runtime, called from a plain thread while the runtime keeps
        // running — the agent's shape. A `std::thread::spawn(..).join()` here would park
        // this current-thread runtime and the timer would never fire: the footgun the
        // docs on `tool_server` name.
        let server = tool_server("s", [Box::new(Sleepy) as Box<dyn Tool>]);
        let answered = tokio::task::spawn_blocking(move || server.tools()[0].call(json!({})))
            .await
            .unwrap();
        assert_eq!(answered, Ok("woke".to_owned()));
    }
}
