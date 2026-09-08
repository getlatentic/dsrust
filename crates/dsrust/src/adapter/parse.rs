//! Reading a model's reply back into the signature's output fields.
//!
//! Each wire format has its own reader: marker sections for `ChatAdapter`, a JSON object for
//! `JsonAdapter`. Both are lenient about what surrounds the answer — prose, code fences,
//! unknown headers — and strict about the answer itself, since a reply that does not speak
//! the format at all is a parse failure rather than a guess.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use crate::signature::{FieldKind, OutField, Signature, annotation};

pub(crate) mod repair;

/// DSPy ChatAdapter's parser: split the reply into sections at `[[ ## name ## ]]` headers,
/// keep the first section seen for each declared output field, ignore prose outside any
/// section and unknown headers (`completed` among them).
/// dspy 3.3.1's `apply_output_field_defaults`: the parsed fields in signature order, with a field
/// the reply left out filled in when the signature lets it be. A declared default answers first,
/// then an annotation that admits `None`; a field with neither stays missing and the caller refuses
/// the reply. The fallback is inserted as it was declared and is not cast against the field, which
/// is what upstream does, so a default the annotation would reject still stands.
///
/// Every adapter's `parse` ends here, so a `Prediction` carries its fields in the order the
/// signature declares them rather than the order the model wrote them.
pub(crate) fn apply_output_field_defaults(
    signature: &Signature,
    fields: Map<String, Value>,
) -> Map<String, Value> {
    let mut completed = Map::new();
    for field in &signature.outputs {
        if let Some(value) = fields.get(&field.name) {
            completed.insert(field.name.clone(), value.clone());
        } else if let Some(default) = &field.default {
            completed.insert(field.name.clone(), default.clone());
        } else if field.allows_none() {
            completed.insert(field.name.clone(), Value::Null);
        }
    }
    completed
}

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
    let fields = apply_output_field_defaults(signature, fields);
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
            signature: signature.clone(),
            message: None,
            reports_parsed: true,
        }));
    }
    // dspy's `ChatAdapter.parse` casts every section with `parse_value(v, annotation)` and raises
    // `AdapterParseError` when a value will not fit its declared type — so a good `int` comes back
    // as `7` rather than `"7"`, and `score: int` answered `very high` is a *parse* failure rather
    // than a validation one. Which matters beyond the value's shape: `ChatAdapter.__call__` answers
    // any exception by re-asking through `JSONAdapter`, so upstream a bad number switches adapters.
    //
    // The same `AdapterParseError` carries both refusals upstream — a missing field and an
    // uncastable one — which is why both are a `FieldMismatch` here, with the cast's own complaint
    // in `message`. The partial travels with it, so a caller who asked for the feedback ask still
    // gets the fields that did read.
    let mut parsed = Value::Object(fields);
    if let Err(error) = signature.coerce(&mut parsed) {
        return Err(anyhow::Error::new(FieldMismatch {
            parsed,
            adapter_name: "ChatAdapter".to_owned(),
            lm_response: raw.to_owned(),
            expected_fields: signature.outputs.iter().map(|f| f.name.clone()).collect(),
            signature: signature.clone(),
            message: Some(error.to_string()),
            // Upstream raises inside its cast loop, before `parsed_result` exists.
            reports_parsed: false,
        }));
    }
    Ok(parsed)
}

/// A section's text as the value it denotes. dspy runs every section through json-repair
/// before validating it, so a `Json` field answered in Python's literal syntax — single
/// quotes, `True`/`False`/`None`, digit-group underscores — lands as its declared type
/// rather than as the text that spells it. Every other section stays text here and is cast by
/// `Signature::coerce` on the way out of [`parse_markers`], which is where a value that will not
/// fit its declared type becomes a parse failure as upstream's does.
pub(super) fn section_value(field: &OutField, text: &str) -> Value {
    match field.kind {
        // `parse_value`'s order for a non-`str` annotation, and the order matters: json-repair
        // first, and Python's own literal syntax only where json-repair answered with the empty
        // string — which is how it reports having found nothing. `'a'` is the case that separates
        // them, since a bare quoted string at the top level is a literal and not a JSON value.
        // A union naming both `None` and `str` is the one shape `parse_value` hands to pydantic
        // whole, so the text stands as itself and never reaches json-repair.
        FieldKind::Json(ref json) if annotation::union_takes_text(&json.annotation) => {
            Value::from(text)
        }
        FieldKind::Json(ref json) => {
            let candidate = repair::loads(text).unwrap_or_else(|_| Value::from(""));
            let candidate = match candidate == "" && !text.is_empty() {
                true => repair::python_literal(text).unwrap_or_else(|| Value::from(text)),
                false => candidate,
            };
            // dspy retries a `dspy.Type` from the original text when the repaired candidate does
            // not validate. `Code` takes a string or a `{"code": ...}` mapping and nothing else,
            // so `int[] a = {1, 9};` — which json-repair reads as `[1, 9]` — stays the code it is.
            match is_code(&json.annotation) && !code_mapping(&candidate) {
                true => Value::from(text),
                false => candidate,
            }
        }
        _ => Value::from(text),
    }
}

/// The annotation of a `Code` field: `Code`, or `Code_java` for dspy's `Code["java"]`.
fn is_code(annotation: &str) -> bool {
    annotation == "Code" || annotation.starts_with("Code_")
}

/// The one repaired shape `Code` accepts: a mapping whose `code` is a string.
fn code_mapping(candidate: &Value) -> bool {
    candidate.get("code").is_some_and(Value::is_string)
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

/// A reply that read as JSON but did not carry the fields the signature declared.
///
/// dspy reports this separately from a reply it could not read at all, and hands the caller
/// whichever declared fields it did find — a partial answer says more about what went wrong
/// than a bare failure does.
#[derive(Debug)]
pub struct FieldMismatch {
    /// The declared fields the reply did carry, in signature order.
    ///
    /// [`Value::Null`] is upstream's `parsed_result=None`, which omits the trailing line — its
    /// guard is `is not None`, so an *empty* object still prints `[]`.
    pub parsed: Value,
    /// dspy's `adapter_name`: which wire format was reading. Empty where the caller did not say.
    pub adapter_name: String,
    /// The reply as it arrived — or, once the JSON adapter's brace search has fired, the object it
    /// pulled out, since upstream rebinds `completion` to the match before it reports anything.
    pub lm_response: String,
    /// Every field the signature declared, in order.
    pub expected_fields: Vec<String>,
    /// dspy's optional `message`, written above the rest and separated by a blank line.
    pub message: Option<String>,
    /// The signature that was being read, which is dspy's `AdapterParseError.signature`.
    ///
    /// The names alone are in `expected_fields`; this is the whole thing, because the one caller
    /// that needs it asks a question names cannot answer. `bootstrap_trace_data` finds the
    /// predictor whose signature *is* this one — `pred.signature == failed_signature` — so it can
    /// record which predictor failed, and two predictors can declare the same field names.
    pub signature: crate::signature::Signature,
    /// Whether upstream would have had a `parsed_result` to report at all.
    ///
    /// False for a cast failure: upstream raises inside its cast loop, before the result is
    /// assembled, so its message stops at the expected-fields line. This crate has the partial in
    /// hand either way and still hands it to a feedback retry, which is why the two are separate
    /// — reporting the partial *and* omitting the line are different questions.
    pub reports_parsed: bool,
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
        if let Some(message) = &self.message {
            write!(out, "{message}\n\n")?;
        }
        write!(
            out,
            "Adapter {} failed to parse the LM response. \n\nLM Response: {} \n\n\
             Expected to find output fields in the LM response: [{}] \n\n",
            self.adapter_name,
            self.lm_response,
            self.expected_fields.join(", "),
        )?;
        // Upstream's guard is `if parsed_result is not None`, so an empty parse still ends with
        // `[]` — it looks like a bug and it is on the wire. `Value::Null` is the `None` that omits
        // it, which upstream reaches only from the field-cast branch this crate casts elsewhere.
        if let Some(parsed) = self.parsed.as_object().filter(|_| self.reports_parsed) {
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
    let kept = apply_output_field_defaults(signature, kept);
    match kept.len() == signature.outputs.len() {
        true => Ok(Value::Object(kept)),
        false => Err(anyhow::Error::new(FieldMismatch {
            parsed: Value::Object(kept),
            adapter_name: adapter_name.to_owned(),
            lm_response: raw.to_owned(),
            expected_fields: signature.outputs.iter().map(|f| f.name.clone()).collect(),
            signature: signature.clone(),
            message: None,
            reports_parsed: true,
        })),
    }
}

/// The object a JSON reply carries, and the text any later failure should name.
///
/// Upstream's recovery is `completion = match.group(0)` — it *rebinds* the variable, so once the
/// brace search has fired every `AdapterParseError` reports the extracted object rather than the
/// reply it came out of. The second half of the pair is that rebinding.
pub(super) fn parse_json<'a>(signature: &Signature, raw: &'a str) -> Result<(Value, &'a str)> {
    // `JSONAdapter.parse` opens with `json_repair.loads(completion)` and then asks
    // `isinstance(fields, dict)` — a reply that reads as an *array* or a scalar is not an answer,
    // so it falls through to the brace search rather than being handed on. `[{"answer": "Paris"}]`
    // is a real reply shape and reaches the object that way.
    if let Ok(value) = repair::loads(raw)
        && value.is_object()
    {
        return Ok((value, raw));
    }
    // The rebinding stands even when the extracted text is still not an object, because upstream
    // assigns before it re-tests — so the refusal names the extract either way.
    let named = first_balanced_braces(raw).unwrap_or(raw);
    if let Ok(value) = repair::loads(named)
        && value.is_object()
    {
        return Ok((value, named));
    }
    Err(anyhow::Error::new(FieldMismatch {
        parsed: Value::Null,
        adapter_name: "JSONAdapter".to_owned(),
        lm_response: named.to_owned(),
        expected_fields: signature.outputs.iter().map(|f| f.name.clone()).collect(),
        message: Some("LM response cannot be serialized to a JSON object.".to_owned()),
        reports_parsed: true,
        signature: signature.clone(),
    }))
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

    /// dspy fills a missing output field from its declared default first and only then from a
    /// `None` the annotation admits, so a field carrying both takes the default. Measured against
    /// `test_missing_optional_output_fields_fall_back_to_defaults`, whose `note` declares both.
    #[test]
    fn a_declared_default_answers_before_the_none_an_annotation_admits() {
        let optional = |name: &str, default: Option<Value>| OutField {
            name: name.into(),
            kind: FieldKind::Json(crate::signature::JsonType::plain(
                "UnionType[str, NoneType]",
            )),
            default,
            ..Default::default()
        };
        let signature = Signature::single_input(
            "Answer.",
            vec![
                optional("note", Some(json!("No note"))),
                optional("maybe", None),
                OutField {
                    name: "answer".into(),
                    ..Default::default()
                },
            ],
        );

        let filled = apply_output_field_defaults(
            &signature,
            [("answer".to_owned(), json!("42"))].into_iter().collect(),
        );
        assert_eq!(filled.get("note"), Some(&json!("No note")));
        assert_eq!(filled.get("maybe"), Some(&Value::Null));

        // A reply that names the field overrides both fallbacks.
        let spoken = apply_output_field_defaults(
            &signature,
            [("note".to_owned(), json!("hello"))].into_iter().collect(),
        );
        assert_eq!(spoken.get("note"), Some(&json!("hello")));

        // A field with neither fallback is left missing for the caller to refuse.
        assert!(!filled.contains_key("answer") || filled["answer"] == json!("42"));
        assert_eq!(spoken.get("answer"), None);
    }

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
        let signature = Signature::single_input(
            "Score it.",
            vec![OutField {
                name: "scores".into(),
                kind: FieldKind::opaque_json(),
                schema: Some(
                    json!({ "type": "object", "additionalProperties": { "type": "number" } }),
                ),
                ..Default::default()
            }],
        );
        let raw = "[[ ## scores ## ]]\n{'score': 123_456.789}\n[[ ## completed ## ]]";
        let value = parse_markers(&signature, raw).expect("parses");
        assert_eq!(value["scores"], json!({ "score": 123_456.789 }));
        assert_eq!(value["scores"]["score"], json!(123456.789));
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
        let (bare, named) = parse_json(&signature(), r#"{ "color": "red" }"#).expect("bare");
        assert_eq!(bare["color"], "red");
        assert_eq!(named, r#"{ "color": "red" }"#, "nothing was extracted");
        // json-repair reads prose and fences itself, so this never reaches the brace search and
        // upstream never rebinds — the reply is what a later failure would name.
        let fenced = "Here it is:\n```json\n{ \"color\": \"blue\" }\n```";
        let (wrapped, named) = parse_json(&signature(), fenced).expect("wrapped");
        assert_eq!(wrapped["color"], "blue");
        assert_eq!(named, fenced);
        assert!(parse_json(&signature(), "no json here").is_err());
    }

    /// The brace search fires only where json-repair answered with something that is not an object,
    /// and upstream's `completion = match.group(0)` rebinds what every later failure reports.
    /// Measured: `dspy.JSONAdapter().parse` writes `LM Response: {"color": "blue"}` for this input,
    /// not the array it arrived in.
    #[test]
    fn the_brace_search_rebinds_what_a_failure_names() {
        let (value, named) =
            parse_json(&signature(), r#"[{"color": "blue"}] trailing"#).expect("extracted");
        assert_eq!(value["color"], "blue");
        assert_eq!(named, r#"{"color": "blue"}"#);
    }

    /// Upstream raises the same `AdapterParseError` with a `message=` prefix here, where the crate
    /// used to answer a bare "model returned invalid JSON".
    #[test]
    fn a_reply_that_is_not_an_object_refuses_the_way_dspy_refuses() {
        let error = parse_json(&signature(), "[1, 2]").expect_err("an array is not an answer");
        assert_eq!(
            error.to_string(),
            "LM response cannot be serialized to a JSON object.\n\nAdapter JSONAdapter failed to \
             parse the LM response. \n\nLM Response: [1, 2] \n\nExpected to find output fields in \
             the LM response: [color, why] \n\n"
        );
    }
}

#[cfg(test)]
mod code_sections {
    use serde_json::json;

    use super::section_value;
    use crate::signature::{FieldKind, JsonType, OutField};

    fn code_field(annotation: &str) -> OutField {
        OutField {
            name: "code".to_owned(),
            kind: FieldKind::Json(JsonType::plain(annotation)),
            ..OutField::default()
        }
    }

    /// The `union_takes_text` guard, which nothing reached through `section_value`: replacing it
    /// with `false` sent a `str | None` field through json-repair and no test noticed.
    ///
    /// `42` is the case that separates the two paths. Upstream hands a union naming both `None`
    /// and `str` to pydantic whole, so the text validates as the string it already is; repaired,
    /// it would arrive as a number and a caller reading `as_str` would find nothing.
    #[test]
    fn a_union_naming_none_and_str_keeps_its_text_rather_than_being_repaired() {
        for annotation in [
            "UnionType[str, NoneType]",
            "typing.Optional[str]",
            "UnionType[NoneType, str]",
        ] {
            assert_eq!(
                section_value(&code_field(annotation), "42"),
                json!("42"),
                "{annotation} takes the text as itself"
            );
        }
    }

    /// The other side of the same guard, so a mutant cannot pass by taking *every* field's text
    /// as itself: a union without `str` is repaired, and `42` arrives as a number.
    #[test]
    fn a_union_without_str_is_repaired_into_its_type() {
        assert_eq!(
            section_value(&code_field("UnionType[int, NoneType]"), "42"),
            json!(42)
        );
    }

    /// Recorded from `adapters/types/code.py`'s docstring example in dsrust-examples: dspy answers
    /// the Java line as code, where json-repair alone answers `[1, 9]`.
    #[test]
    fn a_code_section_json_repair_misreads_stays_text() {
        assert_eq!(
            section_value(&code_field("Code_java"), "int[] a = {1, 9};"),
            json!("int[] a = {1, 9};")
        );
        assert_eq!(
            section_value(&code_field("Code"), "x = [1, 2]"),
            json!("x = [1, 2]")
        );
    }

    #[test]
    fn a_code_mapping_is_still_read_as_one() {
        assert_eq!(
            section_value(&code_field("Code"), r#"{"code": "x = 1"}"#),
            json!({ "code": "x = 1" })
        );
    }
}
