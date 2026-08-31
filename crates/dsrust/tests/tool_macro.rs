//! `#[tool]` against the two things dspy reads off a callable: its docstring and its type hints.

use std::sync::{Arc, Mutex};

use dsrust::{Tool, tool};
use serde_json::json;

// The doc comment is the description, normalised the way a Python docstring is.
//
// dspy runs `inspect.cleandoc` over `__doc__`, so rustdoc's leading space comes off every
// continuation line. " Call once" and "Call once" are different strings in front of the model.
#[tool]
/// Set the learner-facing section title.
///
/// Call once, before writing blocks.
fn set_title(title: String) -> anyhow::Result<String> {
    Ok(format!("Title set to {title:?}."))
}

#[test]
fn the_doc_comment_becomes_the_description() {
    assert_eq!(SetTitle.name(), "set_title");
    assert_eq!(
        SetTitle.description(),
        "Set the learner-facing section title.\n\nCall once, before writing blocks."
    );
}

/// The function is still a function: the attribute adds a tool beside it rather than consuming it,
/// so the logic can be tested without going through a JSON envelope.
#[test]
fn the_function_stays_callable_as_itself() {
    assert_eq!(
        set_title("Fractions".to_owned()).expect("answers"),
        "Title set to \"Fractions\"."
    );
}

// The parameter types become the argument schema, as Python's annotations do.
#[tool]
/// Append one instructional block.
fn add_block(block_type: String, count: u32) -> anyhow::Result<String> {
    Ok(format!("{block_type} x{count}"))
}

#[test]
fn the_parameters_become_the_argument_schema() {
    let args = AddBlock.args();
    assert_eq!(args["block_type"]["type"], "string");
    assert_eq!(args["count"]["type"], "integer");
}

// An argument the model got wrong is *answered*, not raised: a refusal is something the loop can
// read and try again from, where an error ends the turn.
#[tool]
/// Count something.
fn count_it(times: u32) -> anyhow::Result<String> {
    Ok(format!("counted {times}"))
}

#[test]
fn a_bad_argument_is_answered_rather_than_raised() {
    assert_eq!(
        CountIt.call(&json!({ "times": 3 })).expect("answers"),
        "counted 3"
    );
    let wrong = CountIt.call(&json!({ "times": "three" })).expect("answers");
    assert!(
        wrong.starts_with("Refused: `times` is not the type"),
        "{wrong}"
    );
    let missing = CountIt.call(&json!({})).expect("answers");
    assert_eq!(missing, "Refused: this tool needs `times`.");
}

// An argument whose type accepts null is not refused when the model omits it. That is this
// crate's leniency and not a promise upstream makes: dspy exempts an argument from `required`
// for carrying a `default`, never for being nullable, so `Option<T>` alone still asks the model
// for a value — `#[tool(default = ...)]` is what says otherwise. See
// `tests/conformance/react/tool_spec.json`, where `by_type` and `by_default` split on exactly
// this.
#[tool]
/// Add practice, with an optional worked solution.
fn add_practice(prompt: String, worked: Option<String>) -> anyhow::Result<String> {
    Ok(match worked {
        Some(worked) => format!("{prompt} ({worked})"),
        None => prompt,
    })
}

#[test]
fn an_optional_argument_may_be_omitted() {
    assert_eq!(
        AddPractice
            .call(&json!({ "prompt": "Order 1/2 and 1/3." }))
            .expect("answers"),
        "Order 1/2 and 1/3."
    );
}

// A roster over one state. Python captures a draft in a closure; a Rust `fn` captures nothing, so
// the state is the receiver and `&self` is what every tool in the block shares.
struct Draft {
    lines: Mutex<Vec<String>>,
}

#[tool]
impl Draft {
    /// Not marked, so it stays an ordinary method: a roster's constructor is not a tool.
    fn new() -> Self {
        Self {
            lines: Mutex::new(Vec::new()),
        }
    }

    /// Append one line to the draft.
    #[tool]
    fn write(&self, line: String) -> anyhow::Result<String> {
        self.lines.lock().expect("not poisoned").push(line.clone());
        Ok(format!("Wrote {line:?}."))
    }

    /// Read the whole draft back.
    #[tool]
    fn read(&self) -> anyhow::Result<String> {
        Ok(self.lines.lock().expect("not poisoned").join("\n"))
    }
}

#[test]
fn a_roster_shares_one_state() {
    let draft = Arc::new(Draft::new());
    let tools = draft.tools();
    let call = |name: &str, args| {
        tools
            .iter()
            .find(|tool| tool.name() == name)
            .expect("the roster carries it")
            .call(&args)
            .expect("answers")
    };
    call("write", json!({ "line": "first" }));
    call("write", json!({ "line": "second" }));
    assert_eq!(call("read", json!({})), "first\nsecond");
    assert_eq!(
        *draft.lines.lock().expect("not poisoned"),
        ["first", "second"]
    );
}

/// The roster is what the block declares, in the order it declares it — and nothing else, so an
/// unmarked method is not something the model can reach.
#[test]
fn the_roster_is_the_marked_methods_in_order() {
    let draft = Arc::new(Draft::new());
    let tools = draft.tools();
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    assert_eq!(names, ["write", "read"]);
}

/// A roster method's doc comment is its description too, on the same terms.
#[test]
fn a_roster_method_carries_its_doc_comment() {
    let draft = Arc::new(Draft::new());
    let tools = draft.tools();
    assert_eq!(tools[0].description(), "Append one line to the draft.");
    assert_eq!(tools[1].args(), &json!({}));
}

// The block attribute is optional: a method marked on its own gets a tool beside it, named for the
// method. That is the whole difference — `tools()` is what the block form adds.
struct Notes {
    lines: Mutex<Vec<String>>,
}

impl Notes {
    fn new() -> Self {
        Self {
            lines: Mutex::new(Vec::new()),
        }
    }

    /// Write one line down.
    #[tool]
    pub fn note(&self, line: String) -> anyhow::Result<String> {
        self.lines.lock().expect("not poisoned").push(line);
        Ok("Noted.".to_owned())
    }
}

#[test]
fn a_method_is_a_tool_without_the_block_attribute() {
    let notes = Arc::new(Notes::new());
    let tool = notes.note_tool();
    assert_eq!(tool.name(), "note");
    assert_eq!(tool.description(), "Write one line down.");
    assert_eq!(tool.args()["line"]["type"], "string");
    assert_eq!(
        tool.call(&json!({ "line": "first" })).expect("answers"),
        "Noted."
    );
    assert_eq!(*notes.lines.lock().expect("not poisoned"), ["first"]);
    assert_eq!(notes.note("direct".to_owned()).expect("answers"), "Noted.");
}

// dspy's `Tool.acall` awaits a tool whose callable is a coroutine. A tool that reaches a network
// is a future here too, and the agent loops await every tool, so both forms take `async fn`.
#[tool]
/// Fetch what one URL says.
async fn fetch(url: String) -> anyhow::Result<String> {
    tokio::task::yield_now().await;
    Ok(format!("fetched {url}"))
}

struct Remote {
    seen: Mutex<Vec<String>>,
}

impl Remote {
    /// Ask the remote about one term.
    #[tool]
    pub async fn lookup(&self, term: String) -> anyhow::Result<String> {
        tokio::task::yield_now().await;
        self.seen.lock().expect("not poisoned").push(term.clone());
        Ok(format!("looked up {term}"))
    }
}

#[tokio::test]
async fn an_async_free_function_is_a_tool() {
    assert_eq!(Fetch.name(), "fetch");
    assert_eq!(Fetch.description(), "Fetch what one URL says.");
    assert_eq!(Fetch.args()["url"]["type"], "string");
    let answered = Fetch
        .acall_value(&json!({ "url": "https://example.invalid" }))
        .await
        .expect("answers");
    assert_eq!(answered, json!("fetched https://example.invalid"));
    // Bad arguments are still answered rather than raised, on the awaited path too.
    let refused = Fetch.acall_value(&json!({})).await.expect("answers");
    assert_eq!(refused, json!("Refused: this tool needs `url`."));
}

#[tokio::test]
async fn an_async_method_is_a_tool_over_its_receiver() {
    let remote = Arc::new(Remote {
        seen: Mutex::new(Vec::new()),
    });
    let tool = remote.lookup_tool();
    assert_eq!(tool.name(), "lookup");
    assert_eq!(
        tool.acall_value(&json!({ "term": "fractions" }))
            .await
            .expect("answers"),
        json!("looked up fractions")
    );
    assert_eq!(*remote.seen.lock().expect("not poisoned"), ["fractions"]);
}

/// A synchronous tool answers on the awaited path too — upstream's "allow calling a sync tool in
/// the async path", which is what lets a roster mix the two.
#[tokio::test]
async fn a_sync_tool_still_answers_when_awaited() {
    assert_eq!(
        CountIt
            .acall_value(&json!({ "times": 2 }))
            .await
            .expect("answers"),
        json!("counted 2")
    );
}

/// A tool taking its whole argument list as one struct renders each field as the parameter it
/// stands for, which is a different schema from the same field inside a model.
///
/// dspy builds `Tool.args` one parameter at a time — `TypeAdapter(annotation).json_schema()` — so
/// each entry is the root of its own schema: a `str` parameter carries no title, where a `str`
/// *property* of a model is titled after its field name. Rendering the struct in one pass titles
/// every parameter and writes a roster upstream would not.
///
/// Compared as text. Under `preserve_order` two objects differing only in key order are equal, and
/// the key order is what the roster prints.
#[test]
fn a_parameter_is_rendered_as_its_own_root() {
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Origin {
        city: String,
        code: String,
    }
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Args {
        note: String,
        from: Origin,
    }
    let rendered = dsrust::signature::arguments_schema::<Args>().expect("a struct has fields");
    assert_eq!(
        serde_json::to_string(&serde_json::Value::Object(rendered)).expect("serializes"),
        concat!(
            r#"{"note":{"type":"string"},"#,
            r#""from":{"properties":{"city":{"title":"City","type":"string"},"#,
            r#""code":{"title":"Code","type":"string"}},"#,
            r#""required":["city","code"],"title":"Origin","type":"object"}}"#
        )
    );
}

/// Append one instructional block to the draft and return its id.
///
///     block_type is one of: explanation, worked_example.
///     Practice is added with add_practice.
#[tool]
fn append_block(block_type: String, text: String) -> anyhow::Result<String> {
    Ok(format!("{block_type}{text}"))
}

/// A tool's description keeps the shape its author gave it.
///
/// dspy sends `func.__doc__` unnormalised, so a Python tool's description carries whatever
/// indentation the docstring had. A doc comment is this crate's docstring, and running it through
/// the `inspect.cleandoc` a *signature*'s instructions go through would delete the author's
/// indentation rather than an enclosing block's — a Rust doc comment has no enclosing block's
/// indent to find, which is the only thing `cleandoc` exists to remove.
#[test]
fn a_tool_description_keeps_the_indentation_it_was_written_with() {
    assert_eq!(
        AppendBlock.description(),
        "Append one instructional block to the draft and return its id.\n\n    \
         block_type is one of: explanation, worked_example.\n    \
         Practice is added with add_practice.",
    );
}

/// Append one practice question and the answer the learner checks it against.
#[tool]
fn append_practice(
    prompt: String,
    answer: String,
    #[tool(default = "")] worked_solution: String,
) -> anyhow::Result<String> {
    Ok(format!("{prompt}|{answer}|{worked_solution}"))
}

/// A declared default states itself in the schema, and stands in when the model omits it.
///
/// dspy leaves an argument carrying a `default` out of the `required` list it sends the provider,
/// so a Python tool written `worked_solution: str = ""` asks the model for two arguments and this
/// one has to as well. Rust has no default arguments to read, which is why the attribute exists.
#[test]
fn a_tool_argument_can_carry_a_default() {
    assert_eq!(
        serde_json::to_string(AppendPractice.args()).unwrap(),
        r#"{"prompt":{"type":"string"},"answer":{"type":"string"},"worked_solution":{"type":"string","default":""}}"#,
    );
    let answered = AppendPractice
        .call_value(&json!({ "prompt": "2+2?", "answer": "4" }))
        .expect("answers without the defaulted argument");
    assert_eq!(answered, json!("2+2?|4|"));
    // Still refused when an argument with no default is the one missing.
    let refused = AppendPractice
        .call_value(&json!({ "prompt": "2+2?" }))
        .expect("answers");
    assert_eq!(refused, json!("Refused: this tool needs `answer`."));
}

/// Look one term up.
#[tool(
    desc = "Look one term up.\n\n    query is a phrase, not a sentence.\n    Answers with a summary."
)]
fn look_up(query: String) -> anyhow::Result<String> {
    Ok(format!("looked up {query}"))
}

/// A description may be stated outright, which is the only way to write some of them.
///
/// dspy's `Tool` takes `desc` beside the callable for the same reason. Here it also settles a
/// collision Python does not have: rustdoc reads an indented line in a doc comment as a code block
/// and tries to compile and run it, so a description carrying an indented example cannot be
/// written as a doc comment at all.
#[test]
fn a_tool_description_may_be_stated_rather_than_documented() {
    assert_eq!(
        LookUp.description(),
        "Look one term up.\n\n    query is a phrase, not a sentence.\n    Answers with a summary.",
    );
}
