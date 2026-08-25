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

/// dspy `CodeProposalSignature`: what GEPA asks a model when the component is *source*.
///
/// A `Flex`'s optimizable component is a whole `dspy.Module` subclass rather than an instruction, so
/// the proposer is shown the task, the tools in scope, the catalog of primitives it may use, the
/// current source and a batch of failures — and answers with a replacement class.
///
/// The instructions are upstream's docstring, dedented as dspy dedents it, and run to nearly three
/// thousand characters. **Generated from `tests/conformance/code_proposal.json` rather than typed**,
/// and held to it by `every_signature_matches_its_dspy_fixture` — which makes this a drift catcher
/// rather than an independent derivation: it earns its place when the pin moves and the fixture is
/// regenerated, not today.
///
/// Public alongside [`format_failures`] and [`strip_code_fences`], and for their reason: a caller
/// writing their own code proposer needs the prompt dspy sends, not a description of it.
pub fn code_proposal() -> Signature {
    Signature {
        instructions: "Revise the full source code of a dspy.Flex submodule.\n\nYou receive the submodule's task description (its Signature), the available context (any tools\nand style notes), the catalog of allowed primitives, the module's current source, and a batch\nof failing examples with feedback. Produce a revised source that fixes the observed failures\nand follows the catalog.\n\nThe source is ONE ``dspy.Module`` subclass with two coupled methods, and you MUST output the\nentire, internally-consistent class:\n  1. ``def __init__(self):`` calling ``super().__init__()`` and assigning the predictors it\n     needs. Pick the simplest primitive that fits each step: ``dspy.Predict(\"...\")`` for a\n     direct call (the common default), ``dspy.ChainOfThought(\"...\")`` when explicit reasoning\n     helps, and ``dspy.RLM`` / ``dspy.ReAct`` when the step must call tools or explore a\n     large/structured input. Assign no predictors at all if the task needs no LM.\n  2. ``def forward(self, **inputs):`` that calls those predictors as ``self.<name>`` and\n     returns ``dspy.Prediction(<output fields>=...)``.\nBecause ``forward`` calls predictors by name, never rename a predictor in one place without\nupdating the other.\n\nTools are OPTIONAL — use them only when a step genuinely needs one; many good modules are just\na ``dspy.Predict`` or two plus plain Python. When you do use tools, they come from two places.\n(1) Any listed in ``available_context`` are in scope by name — wire the useful ones into\n``dspy.RLM(..., tools=[...])`` / ``dspy.ReAct(..., tools=[...])`` or call them directly\n(reference them by the exact names; do not import or redefine them). If ``available_context``\nis '(no extra context)', no tools were provided — don't reference any.\n(2) AUTHOR your own: when a sub-step needs a capability the provided tools don't cover, define a\ndocumented helper nested inside ``forward`` and call it directly. Authored helpers live in this\nsource, so they are optimized and persisted exactly like the rest of the code. They run in the\nsandbox and cannot be handed to a bridged sub-predictor, so only the provided tools may be wired\ninto ``dspy.RLM``/``dspy.ReAct`` via ``tools=[...]``.\n\nOptimize the predictors' INSTRUCTIONS, not just the code structure. Each predictor's\nnatural-language instructions live in this source — construct a predictor over\n``dspy.Signature(\"inputs -> outputs\", \"instructions\")`` and refine those instructions from the\nfailing examples and feedback (add a clear task definition, domain knowledge the model lacked,\nthe required output format, and rules that prevent the observed errors). These predictors are\ninside a dspy.Flex module, so this source is the ONLY place their prompts get optimized. See the primitives\ncatalog's \"Writing and refining instructions\" section for how. Fix instructions when a failure is about\nWHAT the model should do or know; change the structure when it is about HOW steps are wired.".into(),
        inputs: vec![
            input("task_description", "The submodule's Signature: name, objective, input and output fields."),
            input("available_context", "Tools (in scope by name) and style notes available to the module. May be '(no extra context)'."),
            input("primitives_catalog", "Catalog of allowed primitives and conventions the revised code must follow."),
            input("current_source", "The module's current full source: one dspy.Module subclass (its __init__ and forward)."),
            input("failures", "A batch of failing examples and feedback. Diagnose them and revise the module to fix them."),
        ],
        outputs: vec![
            output("revised_source", "The full revised module source: one `dspy.Module` subclass with `__init__` (predictors, including any refined `dspy.Signature(..., \"instructions\")`) and `forward`."),
        ],
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
/// The prompt is [`code_proposal`], held byte for
/// byte against dspy's. Upstream keeps the current source on any failure that is not an `LMError` —
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
