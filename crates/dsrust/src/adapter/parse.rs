//! Reading a model's reply back into the signature's output fields.
//!
//! Each wire format has its own reader: marker sections for `ChatAdapter`, a JSON object for
//! `JsonAdapter`. Both are lenient about what surrounds the answer — prose, code fences,
//! unknown headers — and strict about the answer itself, since a reply that does not speak
//! the format at all is a parse failure rather than a guess.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use crate::signature::{FieldKind, OutField, Signature};

pub(crate) mod repair;

/// DSPy ChatAdapter's parser: split the reply into sections at `[[ ## name ## ]]` headers,
/// keep the first section seen for each declared output field, ignore prose outside any
/// section and unknown headers (`completed` among them).
pub(super) fn parse_markers(signature: &Signature, raw: &str) -> Result<Value> {
    let mut sections: Vec<(&str, Vec<&str>)> = Vec::new();
    for line in raw.lines() {
        if let Some((name, rest)) = split_header(line) {
            let seed = if rest.is_empty() { vec![] } else { vec![rest] };
            sections.push((name, seed));
        } else if let Some(section) = sections.last_mut() {
            section.1.push(line);
        }
    }
    let mut fields = Map::new();
    for (name, lines) in sections {
        let Some(field) = signature.outputs.iter().find(|field| field.name == name) else {
            continue;
        };
        if fields.contains_key(name) {
            continue;
        }
        let joined = lines.join("\n");
        fields.insert(name.to_owned(), section_value(field, joined.trim()));
    }
    if fields.is_empty() {
        return Err(anyhow!("reply has no [[ ## field ## ]] sections"));
    }
    // dspy's `ChatAdapter.parse` ends on `if fields.keys() != signature.output_fields.keys():
    // raise AdapterParseError`, so a reply short of a field is a *parse* failure and
    // `ChatAdapter.__call__` answers it by re-asking through `JSONAdapter`. Letting it through to
    // validation instead would take a different second ask, with a prompt upstream never sends.
    if fields.len() != signature.outputs.len() {
        return Err(anyhow::Error::new(FieldMismatch {
            parsed: Value::Object(fields),
            adapter_name: "ChatAdapter".to_owned(),
            lm_response: raw.to_owned(),
            expected_fields: signature.outputs.iter().map(|f| f.name.clone()).collect(),
        }));
    }
    // **A known divergence, recorded rather than papered over.** dspy's `ChatAdapter.parse` casts
    // every section with `parse_value(v, annotation)` and raises `AdapterParseError` when a value
    // will not fit, and `ChatAdapter.__call__` answers *any* exception by re-asking through
    // `JSONAdapter`. So upstream a bad number switches adapters. Here it does not: the cast happens
    // in validation, which is what `Predict::feedback_retry` is built on — it re-asks on the same
    // adapter carrying "amount must be a number, got \"abc\"", which upstream has no equivalent of.
    //
    // Calling `coerce_scalars` here makes the parse side match dspy exactly and takes the feedback
    // retry with it, so the two cannot both stand. Filed as `parse-time-casting`; the parse golden
    // records both typed refusals as divergences and goes red when this is resolved.
    Ok(Value::Object(fields))
}

/// A section's text as the value it denotes. dspy runs every section through json-repair
/// before validating it, so a `Json` field answered in Python's literal syntax — single
/// quotes, `True`/`False`/`None`, digit-group underscores — lands as its declared type
/// rather than as the text that spells it. Every other section stays text here and is cast by
/// `coerce_scalars` on the way out of [`parse_markers`], which is where a value that will not fit
/// its declared type becomes a parse failure as upstream's does.
fn section_value(field: &OutField, text: &str) -> Value {
    match field.kind {
        // `parse_value`'s order for a non-`str` annotation, and the order matters: json-repair
        // first, and Python's own literal syntax only where json-repair answered with the empty
        // string — which is how it reports having found nothing. `'a'` is the case that separates
        // them, since a bare quoted string at the top level is a literal and not a JSON value.
        FieldKind::Json(_) => {
            let candidate = repair::loads(text).unwrap_or_else(|_| Value::from(""));
            match candidate == Value::from("") && !text.is_empty() {
                true => repair::python_literal(text).unwrap_or_else(|| Value::from(text)),
                false => candidate,
            }
        }
        _ => Value::from(text),
    }
}

/// Python's `\w`: what `str.isalnum()` accepts, plus `_`.
///
/// Both of dspy's scans over a reply are spelled `\w+` — `\[\[ ## (\w+) ## \]\]` for a marker and
/// `<(?P<name>\w+)>` for a tag — and neither is ASCII. A Python identifier may be any of these, so
/// `réponse` and `答え` are field names dspy renders markers for and reads back. Rust's own
/// `is_alphanumeric` is not this predicate either: it follows `Alphabetic`, which carries combining
/// marks that `str.isalnum()` refuses, so it answers wrongly in the other direction.
fn is_word(letter: char) -> bool {
    json_repair::pychar::is_alnum(letter) || letter == '_'
}

/// A section header at the start of a line: `[[ ## name ## ]]` with a word-character name,
/// keeping any trailing text on the line as that section's first content.
fn split_header(line: &str) -> Option<(&str, &str)> {
    let after_open = line.trim_start().strip_prefix("[[ ## ")?;
    let (name, _) = after_open.split_once(" ## ]]")?;
    if name.is_empty() || !name.chars().all(is_word) {
        return None;
    }
    // dspy matches the header against `line.strip()` and then slices the **unstripped** line at
    // that match's end — `line[match.end():]` — so an indented marker cuts as many characters off
    // the front of what follows as the indent was wide. `    [[ ## answer ## ]]` yields `# ]]`,
    // not an empty rest.
    //
    // That is upstream's off-by-the-indent and it is reproduced rather than corrected, because a
    // model that indents a marker gets these bytes from dspy and must get them here. The offset is
    // in *characters*, as Python's slicing is, not bytes.
    let header = "[[ ## ".len() + name.chars().count() + " ## ]]".len();
    let rest = match line.char_indices().nth(header) {
        Some((at, _)) => &line[at..],
        None => "",
    };
    Some((name, rest.trim()))
}

/// A JSON object anywhere in the reply. Providers in JSON mode return the bare object;
/// models that ignore the mode wrap it in prose or code fences, so the outermost braces
/// are the recovery path (DSPy's JSONAdapter recovers with a regex the same way).
/// Read a reply written as tag pairs.
///
/// dspy scans for `<name>…</name>` over the whole reply and keeps the first occurrence of each
/// declared field, ignoring any tag the signature never asked for. The same mismatch rule as
/// the JSON adapter then applies: a reply missing a declared field is a failure carrying
/// whatever it did say.
pub(super) fn parse_tags(signature: &Signature, raw: &str) -> Result<Value> {
    let mut found = serde_json::Map::new();
    let mut rest = raw;
    while let Some((name, content, after)) = next_tag(rest) {
        if let Some(field) = signature.outputs.iter().find(|field| field.name == name)
            && !found.contains_key(name)
        {
            // Through `section_value` for the same reason the marker path is: dspy hands the body
            // to the field's own Python type, so a structured field written as strict JSON is read
            // as the value it spells rather than kept as the text spelling it.
            found.insert(name.to_owned(), section_value(field, content.trim()));
        }
        rest = after;
    }
    let mut value = declared_fields(signature, Value::Object(found), "XMLAdapter", raw)?;
    // dspy casts each field inside `XMLAdapter.parse` and reports a value that will not fit as
    // a parse failure, rather than handing a caller a string where a number was declared.
    signature
        .coerce_scalars(&mut value)
        .map_err(|error| anyhow!("Failed to parse field in {raw}: {error}"))?;
    Ok(value)
}

/// The next `<name>…</name>` pair: its name, what it wraps, and what follows it.
///
/// A tag name is a word, matching upstream's `\w+`, so punctuation or a space rules a `<`
/// out as an opening tag and the scan moves past it.
fn next_tag(text: &str) -> Option<(&str, &str, &str)> {
    let mut cursor = 0;
    loop {
        let open = text[cursor..].find('<')? + cursor;
        let Some(shut) = text[open..].find('>').map(|at| at + open) else {
            return None;
        };
        let name = &text[open + 1..shut];
        if !name.is_empty() && name.chars().all(is_word) {
            let closing = format!("</{name}>");
            if let Some(end) = text[shut + 1..].find(&closing).map(|at| at + shut + 1) {
                return Some((name, &text[shut + 1..end], &text[end + closing.len()..]));
            }
        }
        cursor = open + 1;
    }
}

/// A reply that read as JSON but did not carry the fields the signature declared.
///
/// dspy reports this separately from a reply it could not read at all, and hands the caller
/// whichever declared fields it did find — a partial answer says more about what went wrong
/// than a bare failure does.
#[derive(Debug)]
pub struct FieldMismatch {
    /// The declared fields the reply did carry, in signature order.
    pub parsed: Value,
    /// dspy's `adapter_name`: which wire format was reading. Empty where the caller did not say.
    pub adapter_name: String,
    /// The reply as it arrived, which is the thing a reader needs to see to fix it.
    pub lm_response: String,
    /// Every field the signature declared, in order.
    pub expected_fields: Vec<String>,
}

impl FieldMismatch {
    /// dspy's `default_code`.
    pub const CODE: &'static str = "adapter_parse_error";
}

impl std::fmt::Display for FieldMismatch {
    /// dspy's `AdapterParseError.__str__`, whitespace included. The trailing space before each
    /// blank line is upstream's and looks accidental; it is on the wire either way, and this is
    /// the text a caller reads when a reply does not parse.
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "Adapter {} failed to parse the LM response. \n\nLM Response: {} \n\n\
             Expected to find output fields in the LM response: [{}] \n\n",
            self.adapter_name,
            self.lm_response,
            self.expected_fields.join(", "),
        )?;
        // Upstream appends this only when something did parse, so an all-or-nothing failure does
        // not end with an empty list that reads like a bug.
        if let Some(parsed) = self.parsed.as_object()
            && !parsed.is_empty()
        {
            let names: Vec<&str> = parsed.keys().map(String::as_str).collect();
            write!(
                out,
                "Actual output fields parsed from the LM response: [{}] \n\n",
                names.join(", ")
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for FieldMismatch {}

/// Keep the declared output fields and fail when the reply did not carry all of them.
///
/// dspy drops anything the signature never asked for, then compares what is left against the
/// declared set — a reply naming only fields the signature does not have is a failure, not an
/// empty success.
pub(super) fn declared_fields(
    signature: &Signature,
    parsed: Value,
    adapter_name: &str,
    raw: &str,
) -> Result<Value> {
    let Some(object) = parsed.as_object() else {
        return Err(anyhow!("model returned invalid JSON"));
    };
    let kept: serde_json::Map<String, Value> = signature
        .outputs
        .iter()
        .filter_map(|field| {
            let value = object.get(&field.name)?;
            Some((field.name.clone(), value.clone()))
        })
        .collect();
    match kept.len() == signature.outputs.len() {
        true => Ok(Value::Object(kept)),
        false => Err(anyhow::Error::new(FieldMismatch {
            parsed: Value::Object(kept),
            adapter_name: adapter_name.to_owned(),
            lm_response: raw.to_owned(),
            expected_fields: signature.outputs.iter().map(|f| f.name.clone()).collect(),
        })),
    }
}

pub(super) fn parse_json(raw: &str) -> Result<Value> {
    // `JSONAdapter.parse` opens with `json_repair.loads(completion)` and then asks
    // `isinstance(fields, dict)` — a reply that reads as an *array* or a scalar is not an answer,
    // so it falls through to the brace search rather than being handed on. `[{"answer": "Paris"}]`
    // is a real reply shape and reaches the object that way.
    if let Ok(value) = repair::loads(raw)
        && value.is_object()
    {
        return Ok(value);
    }
    if let Some(embedded) = first_balanced_braces(raw)
        && let Ok(value) = repair::loads(embedded)
        && value.is_object()
    {
        return Ok(value);
    }
    Err(anyhow!("model returned invalid JSON"))
}

/// The first balanced `{…}` run, which is what dspy's `\{(?:[^{}]|(?R))*\}` finds.
///
/// Not the span from the first `{` to the last `}`: for `{"a": 1} and {"b": 2}` the recursive
/// pattern matches the first object alone, where the outermost span takes both and the prose
/// between them. The scan is blind to quoting, as the pattern is — a `}` inside a string closes
/// the run for both.
fn first_balanced_braces(raw: &str) -> Option<&str> {
    let opens: Vec<usize> = raw.match_indices('{').map(|(at, _)| at).collect();
    for start in opens {
        let mut depth = 0_usize;
        for (offset, letter) in raw[start..].char_indices() {
            match letter {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&raw[start..start + offset + 1]);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::InField;
    use serde_json::json;

    fn signature() -> Signature {
        Signature::single_input(
            "Pick a color.",
            vec![
                OutField {
                    name: "color".into(),
                    desc: "the chosen color".into(),
                    values: Some(vec!["red".into(), "blue".into()]),
                    ..Default::default()
                },
                OutField {
                    name: "why".into(),
                    desc: "one short sentence".into(),
                    ..Default::default()
                },
            ],
        )
    }

    fn json_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Suggest ideas.",
            vec![OutField {
                name: "ideas".into(),
                desc: "three concrete ideas".into(),
                kind: FieldKind::opaque_json(),
                schema: Some(json!({ "type": "array", "items": { "type": "string" } })),
                ..Default::default()
            }],
        );
        signature.inputs = vec![InField {
            name: "recipient".into(),
            desc: "who the gift is for".into(),
            kind: FieldKind::opaque_json(),
            ..Default::default()
        }];
        signature
    }

    #[test]
    fn parse_markers_extracts_fields_and_tolerates_prose() {
        let raw = "Sure, here you go:\n\n[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\nIt is calm.\nVery calm.\n\n[[ ## completed ## ]]\n";
        let value = parse_markers(&signature(), raw).expect("parses");
        assert_eq!(
            value,
            json!({ "color": "red", "why": "It is calm.\nVery calm." })
        );
    }

    #[test]
    fn parse_markers_keeps_first_occurrence_and_same_line_content() {
        let raw = "[[ ## color ## ]] red\n[[ ## color ## ]]\nblue\n[[ ## why ## ]]\ncalm";
        let value = parse_markers(&signature(), raw).expect("parses");
        assert_eq!(value["color"], "red");
    }

    #[test]
    fn parse_markers_refuses_a_reply_short_of_a_field_as_dspy_does() {
        let raw = "[[ ## color ## ]]\nred";
        let refused = parse_markers(&signature(), raw).expect_err("a field is missing");
        let mismatch = refused
            .downcast_ref::<FieldMismatch>()
            .expect("a field mismatch, which is what routes it to the JSON fallback");
        // Whatever did parse rides along, so the fallback's answer can be compared against it.
        assert_eq!(mismatch.parsed, json!({ "color": "red" }));
        assert_eq!(mismatch.adapter_name, "ChatAdapter");
    }

    #[test]
    fn parse_markers_rejects_a_reply_with_no_sections() {
        assert!(parse_markers(&signature(), "red, because it is calm").is_err());
    }

    /// The other direction of the same predicate, and it loses a field rather than a marker.
    ///
    /// A `<…>` the scan calls a tag is consumed whole, so the scan resumes *after* its closing tag
    /// and never looks inside. `char::is_alphanumeric` follows `Alphabetic` and accepts a combining
    /// mark that `str.isalnum()` refuses, which makes `<xֺ>` a tag here and not in dspy — and the
    /// `<answer>` it wraps is skipped with it. Measured: `dspy.XMLAdapter().parse` returns Paris.
    #[test]
    fn a_tag_python_would_not_call_a_tag_does_not_swallow_the_one_inside_it() {
        let signature = Signature::single_input(
            "Answer.",
            vec![OutField {
                name: "answer".into(),
                ..Default::default()
            }],
        );
        let raw = "<x\u{5b0}><answer>Paris</answer></x\u{5b0}>";
        let value = parse_tags(&signature, raw).expect("dspy reads the inner tag");
        assert_eq!(value, json!({ "answer": "Paris" }));
    }

    /// Upstream's header pattern is `\[\[ ## (\w+) ## \]\]`, and Python's `\w` is every code point
    /// `str.isalnum()` accepts plus `_` — not ASCII. A Python identifier may be non-ASCII, so
    /// `réponse` and `答え` are field names dspy renders markers for and parses back, measured
    /// against `dspy.ChatAdapter().parse` on the pin.
    #[test]
    fn a_marker_names_a_field_the_way_python_spells_an_identifier() {
        let signature = Signature::single_input(
            "Answer.",
            vec![
                OutField {
                    name: "réponse".into(),
                    ..Default::default()
                },
                OutField {
                    name: "答え".into(),
                    ..Default::default()
                },
            ],
        );
        let raw = "[[ ## réponse ## ]]\nParis\n\n[[ ## 答え ## ]]\nはい\n\n[[ ## completed ## ]]\n";
        let value = parse_markers(&signature, raw).expect("dspy parses this");
        assert_eq!(value, json!({ "réponse": "Paris", "答え": "はい" }));
    }

    /// Upstream's `test_chat_adapter_parses_float_with_underscores` sends exactly this reply
    /// for a field declared as a model with one float, and expects 123456.789.
    #[test]
    fn the_brace_search_finds_what_dspys_recursive_pattern_finds() {
        // Checked case by case against `regex.search(r"\{(?:[^{}]|(?R))*\}", …)` on the pinned
        // dspy. Four of these are the reason the search is not `find('{')..rfind('}')`: the
        // pattern backtracks past a `{` that never balances, it stops at the *first* complete
        // object rather than spanning to the last brace in the reply, and it is blind to quoting.
        for (raw, expected) in [
            (r#"{"a": 1} and {"b": 2}"#, Some(r#"{"a": 1}"#)),
            ("x {a{b}c} y", Some("{a{b}c}")),
            ("{a{b}", Some("{b}")),
            ("{{a}", Some("{a}")),
            (r#"{"a": "}"}"#, Some(r#"{"a": "}"#)),
            (r#"text {"a": {"b": 1}} tail"#, Some(r#"{"a": {"b": 1}}"#)),
            ("a } b { \"c\": 2 }", Some("{ \"c\": 2 }")),
            (r#"{"a": 1"#, None),
            ("no braces", None),
        ] {
            assert_eq!(first_balanced_braces(raw), expected, "for {raw:?}");
        }
    }

    #[test]
    fn parse_markers_reads_a_json_field_written_as_a_python_literal() {
        let raw = "[[ ## ideas ## ]]\n{'score': 123_456.789}\n[[ ## completed ## ]]";
        let value = parse_markers(&json_signature(), raw).expect("parses");
        assert_eq!(value["ideas"], json!({ "score": 123_456.789 }));
        assert_eq!(value["ideas"]["score"], json!(123456.789));
    }

    #[test]
    fn parse_markers_reads_a_strict_json_field_as_the_value_it_spells() {
        let raw = "[[ ## ideas ## ]]\n[\"a\", \"b\"]";
        let value = parse_markers(&json_signature(), raw).expect("parses");
        // dspy hands the section to the field's own Python type, which reads this as a list, and
        // the parse golden records that. This used to assert the text instead, on the reasoning
        // that a structured field should be left for the caller's typing to judge — true of text
        // that only *might* be JSON, and not of text that is.
        assert_eq!(value["ideas"], json!(["a", "b"]));
    }

    #[test]
    fn parse_json_accepts_bare_and_prose_wrapped_objects() {
        let bare = parse_json(r#"{ "color": "red" }"#).expect("bare");
        assert_eq!(bare["color"], "red");
        let wrapped =
            parse_json("Here it is:\n```json\n{ \"color\": \"blue\" }\n```").expect("wrapped");
        assert_eq!(wrapped["color"], "blue");
        assert!(parse_json("no json here").is_err());
    }
}
