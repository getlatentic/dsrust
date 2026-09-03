//! dspy `utils/inspect_history.py`: the last few calls, rendered for reading.
//!
//! `dspy.inspect_history()` prints; this answers with the text instead, which is the same choice
//! upstream already offers — passing `file=` takes its no-colour path, and a caller who wants it on
//! their terminal writes it there. A library that printed would be deciding what its caller's
//! output looks like.
//!
//! Everything else is upstream's, down to the blank lines: three at the top of every entry and
//! three at the end of the whole dump, `.lstrip()` on the green text and `.strip()` on the rest, a
//! base64 image reduced to the length of its payload, and a trailing completion count written
//! without a newline after it.
//!
//! Held to `history/inspect_history.json`, which records both colour settings from the same call.

use serde_json::Value;

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const BLUE: &str = "\x1b[34m";
const OFF: &str = "\x1b[0m";

/// dspy `pretty_print_history(history, n)`: the last `n` entries as text.
///
/// Each entry is a call as dspy records one — an OpenAI-shaped `messages` list, the `outputs` it
/// answered with, and the time it was made. `colours` is upstream's `use_colors`, which it sets from
/// whether it is writing to a terminal.
///
/// ```
/// use dsrust::lm::api::pretty_print_history;
/// use serde_json::json;
///
/// let history = vec![json!({
///     "messages": [{"role": "user", "content": "  What is 2+2?  "}],
///     "outputs": ["  4  "],
///     "timestamp": "2026-01-01T00:00:00Z",
/// })];
/// let text = pretty_print_history(&history, 1, false);
/// assert!(text.contains("User message:"));
/// // Both sides are stripped, which is why the prompt's own padding is gone.
/// assert!(text.contains("What is 2+2?\n"));
/// ```
pub fn pretty_print_history(history: &[Value], n: usize, colours: bool) -> String {
    let mut out = String::new();
    let start = history.len().saturating_sub(n);
    for item in &history[start..] {
        entry(&mut out, item, colours);
    }
    // Upstream's closing `print("\n\n\n")`, which is four newlines once the print's own is counted.
    out.push_str("\n\n\n\n");
    out
}

fn entry(out: &mut String, item: &Value, colours: bool) {
    out.push_str("\n\n\n\n");
    let timestamp = item
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or("Unknown time");
    line(out, &blue(&format!("[{timestamp}]"), colours));

    // An entry with no `messages` is rendered as its `prompt` in a single user turn.
    let standin;
    let messages: &[Value] = match item.get("messages").and_then(Value::as_array) {
        Some(messages) => messages,
        None => {
            standin = [serde_json::json!({
                "role": "user",
                "content": item.get("prompt").cloned().unwrap_or(Value::Null),
            })];
            &standin
        }
    };
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        line(
            out,
            &red(&format!("{} message:", capitalised(role)), colours),
        );
        content(out, message.get("content").unwrap_or(&Value::Null), colours);
        tool_calls(out, message.get("tool_calls"), colours);
        line(out, "\n");
    }

    let outputs = item
        .get("outputs")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if let Some(first) = outputs.first() {
        response(out, first, colours);
    }
    if outputs.len() > 1 {
        // No newline after this one: upstream passes `end=""` to the colouring helper, so the
        // count runs into whatever the next `print` writes.
        out.push_str(&red_bare(
            &format!(" \t (and {} other completions)", outputs.len() - 1),
            colours,
        ));
        out.push('\n');
    }
}

fn response(out: &mut String, first: &Value, colours: bool) {
    match first.as_object() {
        // A dict output prints its text only when there is some, and then its tool calls.
        Some(fields) => {
            if let Some(text) = fields.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                line(out, &red("Response:", colours));
                // `.strip()` at the call site, and `_green` lstrips again.
                line(out, &green(text.trim(), colours));
            }
            tool_calls(out, fields.get("tool_calls"), colours);
        }
        None => {
            line(out, &red("Response:", colours));
            line(
                out,
                &green(first.as_str().unwrap_or_default().trim(), colours),
            );
        }
    }
}

/// A message's content: a bare string, or the OpenAI content blocks.
fn content(out: &mut String, content: &Value, colours: bool) {
    if let Some(text) = content.as_str() {
        // A bare `print(text)`, so one newline — not the two a colour helper adds.
        line(out, text.trim());
        return;
    }
    for block in content.as_array().map(Vec::as_slice).unwrap_or_default() {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                line(out, text.trim());
            }
            "image_url" => line(out, &blue(&image(block), colours)),
            "input_audio" => line(out, &blue(&audio(block), colours)),
            "file" | "input_file" => line(out, &blue(&attachment(block, kind), colours)),
            _ => {}
        }
    }
}

/// A data URL is reduced to the length of its payload; anything else is shown whole.
fn image(block: &Value) -> String {
    let url = block
        .get("image_url")
        .and_then(|image| image.get("url"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    match url.split_once("base64,") {
        Some((prefix, payload)) => {
            format!("<{prefix}base64,<IMAGE BASE 64 ENCODED({})>", payload.len())
        }
        None => format!("<image_url: {url}>"),
    }
}

fn audio(block: &Value) -> String {
    let audio = block.get("input_audio");
    let format = audio
        .and_then(|a| a.get("format"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let length = audio
        .and_then(|a| a.get("data"))
        .and_then(Value::as_str)
        .map_or(0, str::len);
    format!("<audio format='{format}' base64-encoded, length={length}>")
}

fn attachment(block: &Value, kind: &str) -> String {
    let info = block.get(kind).or_else(|| block.get("file"));
    let field = |name: &str| {
        info.and_then(|i| i.get(name))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    format!(
        "<file: name:{}, id:{}, data_length:{}>",
        field("filename"),
        field("file_id"),
        field("file_data").len()
    )
}

fn tool_calls(out: &mut String, calls: Option<&Value>, colours: bool) {
    let calls = match calls.and_then(Value::as_array) {
        Some(calls) if !calls.is_empty() => calls,
        // Upstream prints the heading for anything truthy, so an empty list prints nothing.
        _ => return,
    };
    line(out, &red("Tool calls:", colours));
    for call in calls {
        let function = call.get("function");
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .or_else(|| call.get("name").and_then(Value::as_str))
            .unwrap_or("<unknown>");
        line(
            out,
            &green(&format!("{name}: {}", arguments(call)), colours),
        );
    }
}

/// The arguments as upstream renders them: `function.arguments` if there is one, else the call's
/// own `args` or `arguments`, parsed from JSON when it is a string that parses.
fn arguments(call: &Value) -> String {
    let raw = call
        .get("function")
        .and_then(|f| f.get("arguments"))
        .or_else(|| call.get("args"))
        .or_else(|| call.get("arguments"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let parsed = match raw.as_str() {
        Some(text) => serde_json::from_str::<Value>(text).unwrap_or(raw.clone()),
        None => raw,
    };
    match parsed.is_object() || parsed.is_array() {
        // `json.dumps(..., ensure_ascii=False)`, which is what this crate's own writer produces.
        true => crate::adapter::python_json::json_dumps(&parsed),
        false => crate::python::text(&parsed),
    }
}

fn capitalised(role: &str) -> String {
    // Python's `str.capitalize`: the first character upper, the rest lower.
    let mut chars = role.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}

/// Upstream's `print(text, file=out)` — the text and a newline.
fn line(out: &mut String, text: &str) {
    out.push_str(text);
    out.push('\n');
}

/// `_green`, whose own `lstrip` is not reproduced: both of its callers strip the value before
/// handing it over, so the lstrip can never see leading space and a `trim_start` here would be a
/// claim no case could ever exercise.
fn green(text: &str, colours: bool) -> String {
    wrapped(GREEN, text, colours)
}

fn red(text: &str, colours: bool) -> String {
    wrapped(RED, text, colours)
}

fn blue(text: &str, colours: bool) -> String {
    wrapped(BLUE, text, colours)
}

/// `_red(..., end="")`: the same colouring with no trailing newline of its own.
fn red_bare(text: &str, colours: bool) -> String {
    match colours {
        true => format!("{RED}{text}{OFF}"),
        false => text.to_owned(),
    }
}

fn wrapped(colour: &str, text: &str, colours: bool) -> String {
    match colours {
        true => format!("{colour}{text}{OFF}\n"),
        false => format!("{text}\n"),
    }
}
