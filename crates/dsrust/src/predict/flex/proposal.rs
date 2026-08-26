//! What GEPA sends when the component it is optimizing is *source*.
//!
//! A `Flex`'s optimizable component is a whole `dspy.Module` subclass, so the proposer is shown the
//! failures and answers with a replacement class. Two strings decide whether that conversation is
//! upstream's: how a batch of failures is rendered into the prompt, and how a fenced reply is read
//! back out. Both are held to `tests/conformance/predict/flex.json`.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::example::Example;
use crate::interpreter::python_literal;
use crate::lm::DynChatModel;
use crate::lm::global::context_model;
use crate::module::Module;
use crate::predict::Predict;
use crate::signature::{InField, OutField, Signature};

/// One declared input of a meta-signature. Spelled here rather than shared with MIPROv2's
/// proposers: reaching across for two five-line constructors meant opening a module path, and doing
/// that changed what rustdoc documents for `MIPROv2` — thirty-four items' worth.
fn input(name: &str, desc: &str) -> InField {
    InField {
        name: name.into(),
        desc: desc.into(),
        ..Default::default()
    }
}

fn output(name: &str, desc: &str) -> OutField {
    OutField {
        name: name.into(),
        desc: desc.into(),
        ..Default::default()
    }
}

/// dspy's `PRIMITIVES_CATALOG`: what the code proposer is told it may write.
///
/// Vendored rather than rewritten, for the shim's reason — the code it describes is Python and runs
/// in the sandbox, so this is the text upstream sends and not a translation of it. Held to the pin
/// by `the_vendored_catalog_is_upstreams_own`.
pub const PRIMITIVES_CATALOG: &str = include_str!("primitives_catalog.txt");

/// dspy `CodeProposalSignature`'s docstring, dedented as dspy dedents it.
///
/// Its own file for the reason [`PRIMITIVES_CATALOG`] is: nearly three thousand characters that
/// have to match upstream byte for byte, and a Rust literal holds them behind an escape per
/// newline and quote — which is a transcription hazard on a value no one can read at a glance.
/// `scripts/generate_fixtures.py` writes this beside the fixture, from the same pinned dspy, so
/// regenerating the pin updates both and `every_signature_matches_its_dspy_fixture` catches a
/// stale one.
pub const CODE_PROPOSAL_INSTRUCTIONS: &str = include_str!("code_proposal_instructions.txt");

/// dspy `CodeProposalSignature`: what GEPA asks a model when the component is *source*.
///
/// A `Flex`'s optimizable component is a whole `dspy.Module` subclass rather than an instruction, so
/// the proposer is shown the task, the tools in scope, the catalog of primitives it may use, the
/// current source and a batch of failures — and answers with a replacement class.
///
/// The instructions are [`CODE_PROPOSAL_INSTRUCTIONS`], vendored rather than typed for the reason
/// the catalog above is: they are nearly three thousand characters of upstream's docstring, and a
/// Rust string literal would hold them behind an escape for every newline and quote.
///
/// Public alongside [`format_failures`] and [`strip_code_fences`], and for their reason: a caller
/// writing their own code proposer needs the prompt dspy sends, not a description of it.
pub fn code_proposal() -> Signature {
    Signature {
        instructions: CODE_PROPOSAL_INSTRUCTIONS.to_owned(),
        inputs: vec![
            input(
                "task_description",
                "The submodule's Signature: name, objective, input and output fields.",
            ),
            input(
                "available_context",
                "Tools (in scope by name) and style notes available to the module. May be '(no extra context)'.",
            ),
            input(
                "primitives_catalog",
                "Catalog of allowed primitives and conventions the revised code must follow.",
            ),
            input(
                "current_source",
                "The module's current full source: one dspy.Module subclass (its __init__ and forward).",
            ),
            input(
                "failures",
                "A batch of failing examples and feedback. Diagnose them and revise the module to fix them.",
            ),
        ],
        outputs: vec![output(
            "revised_source",
            "The full revised module source: one `dspy.Module` subclass with `__init__` (predictors, including any refined `dspy.Signature(..., \"instructions\")`) and `forward`.",
        )],
    }
}

/// dspy's `_format_failures`: a batch of failing examples as the prompt shows them.
///
/// The values go through Python's `repr`, not JSON — `{'question': 'Where?'}`, single-quoted, with
/// CPython's quote-switching for an apostrophe. Rendering them as JSON would put a different string
/// in front of the model, which is the same defect [`python_literal`] was written for.
///
/// A record missing a key prints `None`, because upstream reaches for `.get`.
pub fn format_failures(records: &[Map<String, Value>]) -> String {
    if records.is_empty() {
        return "(no failing examples available)".to_owned();
    }
    records
        .iter()
        .enumerate()
        .map(|(at, record)| {
            let shown = |key: &str| record.get(key).unwrap_or(&Value::Null);
            format!(
                "=== Example {at} ===\nInputs:\n{}\nGenerated Outputs:\n{}\nFeedback:\n{}",
                python_literal(shown("Inputs")),
                python_literal(shown("Generated Outputs")),
                // Feedback is interpolated rather than repr'd — upstream's one `!r`-less field.
                match record.get("Feedback") {
                    Some(Value::String(text)) => text.clone(),
                    Some(value) => python_literal(value),
                    None => "None".to_owned(),
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// dspy's `_strip_code_fences`: the source out of a fenced reply.
///
/// The opening line goes whatever it says, so ```` ```python ```` and a bare ```` ``` ```` are the
/// same; a closing fence goes only if it is there. A reply that is nothing but ```` ``` ```` has no
/// newline to cut at and loses its three characters to the closing rule, leaving nothing — which is
/// upstream's behaviour and worth reproducing rather than tidying.
///
/// Tabs expand to four spaces, because the reply is about to be executed and Python refuses mixed
/// indentation.
pub fn strip_code_fences(reply: &str) -> String {
    let mut source = reply.trim().to_owned();
    if source.starts_with("```") {
        if let Some(newline) = source.find('\n') {
            source = source[newline + 1..].to_owned();
        }
        if let Some(without) = source.strip_suffix("```") {
            source = without.to_owned();
        }
    }
    expand_tabs(source.trim())
}

/// Python's `str.expandtabs(4)`: a tab advances to the next multiple of four *within its line*.
///
/// Not four spaces each — a tab after two characters advances two, which is the whole point of the
/// call and the half a `replace('\t', "    ")` would get wrong.
fn expand_tabs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut column = 0;
    for character in text.chars() {
        match character {
            '\t' => {
                let advance = 4 - (column % 4);
                out.extend(std::iter::repeat_n(' ', advance));
                column += advance;
            }
            '\n' => {
                out.push('\n');
                column = 0;
            }
            other => {
                out.push(other);
                column += 1;
            }
        }
    }
    out
}

/// The proposer's inputs for one code component, in the order the signature declares them.
fn proposal_inputs(
    task_description: &str,
    available_context: &str,
    primitives_catalog: &str,
    current_source: &str,
    failures: &[Map<String, Value>],
) -> Example {
    Example::new([
        (
            "task_description",
            Value::String(task_description.to_owned()),
        ),
        (
            "available_context",
            Value::String(available_context.to_owned()),
        ),
        (
            "primitives_catalog",
            Value::String(primitives_catalog.to_owned()),
        ),
        ("current_source", Value::String(current_source.to_owned())),
        ("failures", Value::String(format_failures(failures))),
    ])
}

/// dspy `propose_code`: ask for a revised module for each code component, keeping the original when
/// the proposer cannot answer.
///
/// The prompt is [`code_proposal`], held to dspy's by
/// `the_code_proposal_signature_is_dspys_own` — field for field and byte for byte, which nothing
/// checked until that test was written. Upstream keeps the current source on any failure that is not an `LMError` —
/// a proposal that fails to arrive should cost a generation, not the run — and lets a model failure
/// through, which is the distinction reproduced here: a refusal to *answer* keeps the source, and a
/// refusal to *reach the model* is the caller's to see.
pub async fn propose_code(
    components: &[String],
    candidate: &BTreeMap<String, String>,
    failures: &BTreeMap<String, Vec<Map<String, Value>>>,
    task_descriptions: &BTreeMap<String, String>,
    contexts: &BTreeMap<String, String>,
    primitives_catalog: &str,
    reflection: Arc<dyn DynChatModel>,
) -> BTreeMap<String, String> {
    let proposer = Predict::from_signature(code_proposal());
    let mut proposed = BTreeMap::new();
    for component in components {
        let Some(current) = candidate.get(component) else {
            continue;
        };
        let asked = proposal_inputs(
            task_descriptions
                .get(component)
                .map_or(component.as_str(), String::as_str),
            contexts
                .get(component)
                .map_or("(no extra context)", String::as_str),
            primitives_catalog,
            current,
            failures.get(component).map_or(&[][..], Vec::as_slice),
        );
        let revised = context_model(reqwest::Client::new(), reflection.clone())
            .run(proposer.forward(asked))
            .await
            .ok()
            .and_then(|answered| {
                answered
                    .get("revised_source")
                    .and_then(Value::as_str)
                    .map(strip_code_fences)
            });
        proposed.insert(
            component.clone(),
            revised.unwrap_or_else(|| current.clone()),
        );
    }
    proposed
}

/// dspy `rebind_flex_code`: apply a proposed source to each `Flex` the candidate names.
///
/// A candidate keyed by a path the program has no `Flex` for is ignored, as upstream ignores it —
/// a candidate carries every component and only some of them are code. Source that does not name a
/// class is refused here rather than at the next forward, which is where upstream's `_bind_code`
/// refuses it too.
pub fn rebind_flex_code(
    program: &mut (impl Module + ?Sized),
    candidate: &BTreeMap<String, String>,
) -> Result<()> {
    for named in program.named_flexes() {
        if let Some(source) = candidate.get(&named.name) {
            named.flex.bind(source.clone())?;
        }
    }
    Ok(())
}

/// dspy `enumerate_flex_submodules`: the source each `Flex` in the program currently holds.
///
/// The read half of [`rebind_flex_code`], and what seeds a candidate: an optimizer starts from what
/// the program already says and proposes a replacement.
pub fn flex_components(program: &mut (impl Module + ?Sized)) -> BTreeMap<String, String> {
    program
        .named_flexes()
        .into_iter()
        .map(|named| (named.name, named.flex.module_src().to_owned()))
        .collect()
}

/// dspy `flex_task_context`: what each `Flex` in a program shows the code proposer.
///
/// The two maps [`propose_code`] takes, keyed by the same component path
/// [`flex_components`] keys — the signature spelled out, and the tools in scope.
pub fn flex_task_context(
    program: &mut (impl Module + ?Sized),
) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let mut described = BTreeMap::new();
    let mut contexts = BTreeMap::new();
    for named in program.named_flexes() {
        described.insert(named.name.clone(), named.flex.signature_spec());
        contexts.insert(named.name, named.flex.context_blurb(true));
    }
    (described, contexts)
}
