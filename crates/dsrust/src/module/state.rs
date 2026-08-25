//! A compiled program on disk, in the file dspy writes.
//!
//! This is the artifact an optimizer exists to produce, and it is portable: a program compiled by
//! this crate is saved in exactly the shape `dspy.Module.load` reads, so the JSON can be carried
//! to Python and run there. That is the whole reason the shape is upstream's rather than whatever
//! serde would have written — `tests/saved_program.rs` and the bridge hold it to that claim.
//!
//! What each optimizer leaves here differs, though the shape does not: `BootstrapFewShot` fills
//! `demos` and leaves the instructions alone, while `GEPA` and `COPRO` rewrite
//! `signature.instructions` and leave `demos` empty. Either way the prompt's ingredients are all
//! legible in the file — the assembled prompt is not, because an adapter renders that at call
//! time from these pieces.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::example::Example;
use crate::signature::{Signature, infer_prefix};

/// The dspy release this crate is a port of, stated in every file it writes.
///
/// dspy's `load` compares each version it finds against the running environment and warns where
/// they differ, so this is a claim that has to be true: a file saved here was written to the
/// format that release reads.
pub const DSPY_VERSION: &str = "3.3.0b1";

/// dspy `Predict.dump_state`: everything a saved predictor restores.
///
/// `traces` and `train` are dspy's own vestigial state — always empty in a saved program, and
/// kept because the file is read by a loader that expects the keys.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PredictorState {
    #[serde(default)]
    pub traces: Vec<Value>,
    #[serde(default)]
    pub train: Vec<Value>,
    #[serde(default)]
    pub demos: Vec<Map<String, Value>>,
    pub signature: SignatureState,
    /// The model this predictor was pinned to, as it states itself.
    ///
    /// dspy's `Predict.dump_state`'s `lm` key. `null` for a predictor that was never pinned, and
    /// for one pinned to a model with nothing reconstructible to say — see
    /// [`ChatModel::dump_state`](crate::ChatModel::dump_state). Never carries a credential.
    #[serde(default)]
    pub lm: Option<Value>,
}

/// dspy `Signature.dump_state`: the objective, and each field's prompt-facing description.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignatureState {
    pub instructions: String,
    /// Inputs then outputs, in declaration order — the order dspy's `fields` dict has, and the
    /// order its loader zips these back onto.
    pub fields: Vec<FieldState>,
}

/// One field as a saved program states it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FieldState {
    /// dspy's field prefix — `Question:`. Inferred from the name where nothing set it, which is
    /// what `infer_prefix` is for.
    pub prefix: String,
    /// dspy's field `desc`, which defaults to `${name}` rather than to nothing.
    pub description: String,
}

/// What dspy checks when it opens a saved file.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Metadata {
    pub dependency_versions: BTreeMap<String, String>,
}

impl Default for Metadata {
    /// Only `dspy` is stated. dspy's loader looks up every version it finds in the file against
    /// the running environment, so naming `python` or `cloudpickle` here would either be a guess
    /// about someone else's interpreter or a warning about a library this crate does not use.
    fn default() -> Self {
        Self {
            dependency_versions: BTreeMap::from([("dspy".to_owned(), DSPY_VERSION.to_owned())]),
        }
    }
}

/// What one submodule of a program saved.
///
/// Not every submodule is a predictor. dspy's state map holds whatever each one's `dump_state`
/// returned, and a `dspy.Flex` returns `{module_src, lm}` — no signature, no demos. Running a
/// program holding one of each writes both shapes side by side under a single map, so a map typed
/// to predictor states cannot read its own file back.
///
/// Untagged, because dspy writes no discriminator: the shapes are told apart by what they carry, and
/// a `signature` is what makes a predictor one.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum SubmoduleState {
    Predictor(PredictorState),
    Flex(FlexState),
}

/// dspy `Flex.dump_state`: the source an optimizer rewrote, and the model it was pinned to.
///
/// A `Flex`'s update unit is its source rather than a signature and some demos, so this is the whole
/// of what it saves.
/// `deny_unknown_fields` is what makes the untagged read honest. Both fields are optional, so
/// without it *any* object matches — and a predictor entry that lost its signature would read as a
/// `Flex` holding no source rather than failing, which is a corrupt saved program loading quietly.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlexState {
    pub module_src: Option<String>,
    #[serde(default)]
    pub lm: Option<Value>,
}

/// A whole saved program: each submodule under its own name, and the metadata beside them.
///
/// dspy writes them at the top level rather than under a key, which is why they are flattened here
/// — the file is a map of submodule names with `metadata` among them.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProgramState {
    #[serde(flatten)]
    pub submodules: BTreeMap<String, SubmoduleState>,
    #[serde(default)]
    pub metadata: Metadata,
}

impl ProgramState {
    /// The state for these named submodules, with the metadata dspy expects.
    pub fn new(submodules: BTreeMap<String, SubmoduleState>) -> Self {
        Self {
            submodules,
            metadata: Metadata::default(),
        }
    }

    /// One submodule's state, whatever shape it saved in.
    pub fn state(&self, name: &str) -> Option<&SubmoduleState> {
        self.submodules.get(name)
    }

    /// One predictor's state, or nothing when that name saved something else.
    pub fn get(&self, name: &str) -> Option<&PredictorState> {
        match self.submodules.get(name) {
            Some(SubmoduleState::Predictor(predictor)) => Some(predictor),
            _ => None,
        }
    }
}

impl PredictorState {
    /// What one predictor's signature, demos and pinned model amount to on disk.
    pub fn of(signature: &Signature, demos: &[Example], lm: Option<Map<String, Value>>) -> Self {
        Self {
            traces: Vec::new(),
            train: Vec::new(),
            demos: demos.iter().map(demo_fields).collect(),
            signature: SignatureState::of(signature),
            lm: lm.map(Value::Object),
        }
    }
}

impl SignatureState {
    pub fn of(signature: &Signature) -> Self {
        let inputs = signature
            .inputs
            .iter()
            .map(|field| FieldState::of(&field.name, &field.desc, field.prefix.as_deref()));
        let outputs = signature
            .outputs
            .iter()
            .map(|field| FieldState::of(&field.name, &field.desc, field.prefix.as_deref()));
        Self {
            instructions: signature.instructions.clone(),
            fields: inputs.chain(outputs).collect(),
        }
    }

    /// Write this state back onto a signature, which must be the one it was dumped from.
    ///
    /// dspy zips the saved fields onto the live ones and stops at the shorter, so a file whose
    /// program has since gained a field restores what it can rather than failing.
    pub fn restore(&self, signature: &mut Signature) {
        signature.instructions = self.instructions.clone();
        let inputs = signature
            .inputs
            .iter_mut()
            .map(|field| (&mut field.desc, &mut field.prefix));
        let outputs = signature
            .outputs
            .iter_mut()
            .map(|field| (&mut field.desc, &mut field.prefix));
        for ((desc, prefix), saved) in inputs.chain(outputs).zip(&self.fields) {
            *desc = saved.description.clone();
            *prefix = Some(saved.prefix.clone());
        }
    }
}

impl FieldState {
    fn of(name: &str, desc: &str, prefix: Option<&str>) -> Self {
        Self {
            prefix: prefix.map_or_else(|| format!("{}:", infer_prefix(name)), str::to_owned),
            // dspy's `desc` defaults to the field's own name in placeholder form, and a saved
            // file states that default rather than omitting it.
            description: match desc.is_empty() {
                true => format!("${{{name}}}"),
                false => desc.to_owned(),
            },
        }
    }
}

/// A demo as its ordered field map, for saving; `serde_json`'s `preserve_order` keeps signature
/// order so a reloaded demo renders exactly as it did.
fn demo_fields(demo: &Example) -> Map<String, Value> {
    demo.fields()
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect()
}

/// A saved demo back as an [`Example`], its input split re-declared from the signature — which is
/// where dspy keeps it too, so it need not be stored.
pub(super) fn demo_from_fields(fields: &Map<String, Value>, inputs: &[String]) -> Example {
    Example::new(
        fields
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    )
    .with_inputs(inputs.iter().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{InField, OutField};

    fn signature() -> Signature {
        Signature {
            instructions: "Given the fields `plain_question`, `with_desc`, produce the fields \
                           `the_answer`."
                .into(),
            inputs: vec![
                InField {
                    name: "plain_question".into(),
                    ..Default::default()
                },
                InField {
                    // Named for `infer_prefix`, not for anything this crate calls: a leading
                    // `with_` is what makes the prefix `With Desc:`, so renaming it would delete
                    // the case rather than move it.
                    name: "with_desc".into(),
                    desc: "a described one".into(),
                    ..Default::default()
                },
            ],
            outputs: vec![OutField {
                name: "the_answer".into(),
                desc: "what came out".into(),
                ..Default::default()
            }],
        }
    }

    /// The expected values are dspy 3.3.0b1's own, for the same signature — its prefixes come
    /// from `infer_prefix`, and an undescribed field states `${name}` rather than nothing.
    #[test]
    fn a_signature_dumps_the_way_dspy_dumps_it() {
        let state = SignatureState::of(&signature());
        assert_eq!(
            state.fields,
            vec![
                FieldState {
                    prefix: "Plain Question:".into(),
                    description: "${plain_question}".into()
                },
                FieldState {
                    prefix: "With Desc:".into(),
                    description: "a described one".into()
                },
                FieldState {
                    prefix: "The Answer:".into(),
                    description: "what came out".into()
                },
            ]
        );
    }

    #[test]
    fn a_saved_signature_restores_onto_the_one_it_came_from() {
        let mut edited = signature();
        edited.instructions = "Answer with GOOD precision.".into();
        edited.inputs[0].desc = "the question, rewritten".into();
        let saved = SignatureState::of(&edited);

        let mut restored = signature();
        saved.restore(&mut restored);
        assert_eq!(restored.instructions, "Answer with GOOD precision.");
        assert_eq!(restored.inputs[0].desc, "the question, rewritten");
        assert_eq!(
            restored.inputs[0].prefix.as_deref(),
            Some("Plain Question:")
        );
        assert_eq!(SignatureState::of(&restored), saved);
    }

    /// dspy's loader requires the block and looks up every version named in it, so stating one it
    /// cannot resolve would be worse than stating fewer.
    #[test]
    fn the_metadata_names_the_dspy_release_this_is_a_port_of() {
        let written = serde_json::to_value(Metadata::default()).expect("serializes");
        assert_eq!(
            written,
            serde_json::json!({ "dependency_versions": { "dspy": DSPY_VERSION } })
        );
    }

    /// The predictors sit at the top level beside `metadata`, which is the shape dspy's loader
    /// indexes — not a nested object.
    #[test]
    fn a_program_writes_its_predictors_beside_the_metadata() {
        let state = ProgramState::new(BTreeMap::from([(
            "predict".to_owned(),
            SubmoduleState::Predictor(PredictorState::of(&signature(), &[], None)),
        )]));
        let written = serde_json::to_value(&state).expect("serializes");
        assert!(written["predict"]["signature"]["instructions"].is_string());
        assert!(
            written["predict"]["traces"]
                .as_array()
                .expect("traces")
                .is_empty()
        );
        assert_eq!(written["predict"]["lm"], Value::Null);
        assert!(written["metadata"]["dependency_versions"]["dspy"].is_string());
        assert_eq!(
            serde_json::from_value::<ProgramState>(written).expect("round trips"),
            state
        );
    }
}
