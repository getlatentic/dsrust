use anyhow::{Result, anyhow};
use serde_json::{Map, Value, json};

mod field_type;
mod parse;
mod prefix;
mod reflect;

pub(crate) use field_type::wire_forms;
pub use field_type::{FieldKind, JsonType, LiteralValue, TypeDescription};
pub(crate) use field_type::coerce_value;
use field_type::annotation_of;
pub use parse::parse;
pub use prefix::infer_prefix;
pub use reflect::json_field_reflection;

/// The derive plus its call-site macros: `predict!(Task { field: value, ... })` and the
/// `chain_of_thought!` twin evaluate to one typed module call awaiting the caller's `?`.
pub use dsrust_derive::{Signature, chain_of_thought, predict, signature};

/// One input field of a signature: a name, a one-line description, a wire type, an optional
/// closed set the prompt spells as the field's type in place of that wire type, and the prose
/// its declared constraints read as.
///
/// [`Default`] is what keeps a field cheap to extend: every construction site names only the
/// members it means and takes the rest from here, so a member added later costs no edits.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct InField {
    pub name: String,
    pub desc: String,
    pub kind: FieldKind,
    pub values: Option<Vec<LiteralValue>>,
    /// What pydantic's constraints on this field say, already in prose — `minimum length: 5`,
    /// `greater than or equal to: 0, less than or equal to: 10`.
    ///
    /// dspy computes this string when the signature is declared, so it crosses the bridge as
    /// data and this crate only decides where in the prompt it reads. A signature declared in
    /// Rust has no constraints to state and leaves it empty.
    pub constraints: Option<String>,
    /// dspy's field prefix — `Question:` for a field named `question`. `None` means nobody set
    /// one and it is inferred from the name, which is what dspy does too; a saved program restores
    /// whatever was in force when it was compiled.
    pub prefix: Option<String>,
}

impl InField {
    /// The Python type prompts print for this field. An input carries no schema and yields no
    /// reply to check, so a closed set here is what the model is shown and nothing besides.
    pub fn annotation(&self) -> String {
        annotation_of(self.values.as_ref(), &self.kind)
    }
}

/// One output field of a signature: a name, a one-line description, a wire type, an
/// optional closed set of allowed values (legal on `Str` fields only), — for `Json`
/// fields — the nested JSON schema of the declared type, and the prose its declared
/// constraints read as.
///
/// [`Default`] carries the same weight it does on [`InField`]: name the members that differ,
/// take the rest from here.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct OutField {
    pub name: String,
    pub desc: String,
    pub kind: FieldKind,
    pub values: Option<Vec<LiteralValue>>,
    pub schema: Option<Value>,
    /// What pydantic's constraints on this field say, already in prose. See
    /// [`InField::constraints`].
    pub constraints: Option<String>,
    /// dspy's field prefix — `Question:` for a field named `question`. `None` means nobody set
    /// one and it is inferred from the name, which is what dspy does too; a saved program restores
    /// whatever was in force when it was compiled.
    pub prefix: Option<String>,
}

impl OutField {
    /// The Python type prompts print for this field, a closed set standing in for the kind.
    pub fn annotation(&self) -> String {
        annotation_of(self.values.as_ref(), &self.kind)
    }

    /// The property this field contributes to [`Signature::schema`]: scalar kinds map to
    /// their type name plus any closed set; a `Json` field drops in its real nested schema
    /// so structured-output providers enforce the shape, or accepts anything without one.
    fn property_schema(&self) -> Value {
        let Some(type_name) = self.kind.schema_type() else {
            return self.schema.clone().unwrap_or_else(|| json!({}));
        };
        let mut spec = Map::new();
        let Some(values) = &self.values else {
            spec.insert("type".into(), json!(type_name));
            return Value::Object(spec);
        };
        // A mixed Python `Literal` has no one JSON type to name, so `type` is stated only
        // where it holds of every member. The enum pins the value either way.
        if values.iter().all(|value| value.schema_type() == type_name) {
            spec.insert("type".into(), json!(type_name));
        }
        let members: Vec<Value> = values.iter().map(LiteralValue::to_json).collect();
        spec.insert("enum".into(), json!(members));
        Value::Object(spec)
    }

    /// The one-line shape note prompt surfaces append for `Json` fields, so marker-path
    /// models with no schema support still see the expected structure.
    pub(crate) fn schema_suffix(&self) -> Option<String> {
        let schema = self.schema.as_ref()?;
        Some(format!("json matching schema: {schema}"))
    }
}

/// The nested schema a derived `Json` output carries: fully inlined (no `$ref`) and free of
/// root metadata, so it drops straight into a property slot and stays a one-liner in prompts.
pub fn json_field_schema<T: schemars::JsonSchema>() -> Value {
    let generator = schemars::generate::SchemaSettings::default()
        .with(|settings| {
            settings.inline_subschemas = true;
            settings.meta_schema = None;
        })
        .into_generator();
    let mut root = generator.into_root_schema_for::<T>().to_value();
    if let Some(object) = root.as_object_mut() {
        object.remove("title");
    }
    root
}

/// A DSPy-style signature: the task instructions plus the typed input and output fields.
/// The signature owns WHAT the task is; the modules in [`mod@crate::predict`] own HOW the model
/// is asked.
/// `PartialEq` is dspy's `Signature.equals`, which an optimizer calls to refuse a teacher whose
/// program is not the student's twin. dspy compares instructions and each field's schema notes;
/// the same members are what these carry.
#[derive(Clone, PartialEq, Debug)]
pub struct Signature {
    pub instructions: String,
    pub inputs: Vec<InField>,
    pub outputs: Vec<OutField>,
}

/// A derived signature declaration: the struct carries the field lists at compile time, and
/// the companions carry the values at run time. Implemented by `#[derive(Signature)]`.
pub trait SignatureSpec {
    type Inputs;
    type Outputs: serde::de::DeserializeOwned;
    fn signature() -> Signature;
    /// The input values in signature order, ready for the adapters to render.
    fn input_pairs(inputs: &Self::Inputs) -> Vec<crate::adapter::Input<'static>>;
}

/// dspy's string signature: `"email -> sentiment".parse()`.
///
/// The field names and their order are the declaration; a name with no type is a string, and
/// the instructions read the way upstream writes them when nobody wrote any.
impl std::str::FromStr for Signature {
    type Err = anyhow::Error;

    fn from_str(spelling: &str) -> Result<Self> {
        parse::parse(spelling)
    }
}

impl Signature {
    /// A signature whose whole input is one free-form `request` field, for callers that
    /// render their own prompt string.
    pub fn single_input(instructions: impl Into<String>, outputs: Vec<OutField>) -> Self {
        Self {
            instructions: instructions.into(),
            inputs: vec![InField {
                name: "request".into(),
                desc: "the request".into(),
                ..Default::default()
            }],
            outputs,
        }
    }

    /// dspy `Signature.delete`: this signature without the named field, from whichever side
    /// holds it. A name that is not there leaves the signature unchanged, as upstream has it —
    /// deleting a field an adapter only sometimes adds should not depend on whether it did.
    pub fn delete(&self, name: &str) -> Self {
        let mut without = self.clone();
        without.inputs.retain(|field| field.name != name);
        without.outputs.retain(|field| field.name != name);
        without
    }

    /// dspy `Signature.with_instructions`: the same fields under a different objective. What an
    /// optimizer produces — every proposal it scores is this call.
    pub fn with_instructions(&self, instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            ..self.clone()
        }
    }

    /// JSON schema for providers with native structured output.
    pub fn schema(&self) -> Value {
        let mut properties = Map::new();
        for field in &self.outputs {
            properties.insert(field.name.clone(), field.property_schema());
        }
        let required: Vec<&str> = self.outputs.iter().map(|f| f.name.as_str()).collect();
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    }

    /// Prompt clause carrying the same contract for providers without structured output.
    /// A `Json` field's shape note replaces its kind annotation: the schema already says
    /// "json" and the pair would read twice.
    pub fn output_clause(&self) -> String {
        let keys: Vec<&str> = self.outputs.iter().map(|f| f.name.as_str()).collect();
        let mut clause = format!("Respond with a JSON object with keys {}.", keys.join(", "));
        for field in &self.outputs {
            clause.push(' ');
            clause.push_str(&field.name);
            let suffix = field.schema_suffix();
            if field.kind != FieldKind::Str && suffix.is_none() {
                clause.push_str(&format!(" ({})", field.kind.annotation()));
            }
            clause.push_str(": ");
            clause.push_str(&field.desc);
            if let Some(suffix) = suffix {
                clause.push_str(&format!(" ({suffix})"));
            }
            if let Some(values) = &field.values {
                clause.push_str(&format!(" (one of: {})", wire_forms(values, ", ")));
            }
            clause.push('.');
        }
        clause
    }

    /// Coerce every declared output to its kind, before [`Self::ensure`] checks the reply.
    /// The chat adapter always hands strings; JSON-mode models return native values or
    /// strings depending on the provider, so both spellings parse the same way. A failure
    /// reads as retry feedback the model can act on, like ensure's own errors. Missing
    /// fields are skipped so ensure reports them as missing.
    /// Coerce only the fields whose wire form this crate can read on its own.
    ///
    /// A scalar's is unambiguous: `int` means the text is a number or the reply is wrong, and an
    /// adapter that casts while parsing can say so there. A structured field's is not — the text
    /// may be JSON, or it may be the form its own type accepts, and dspy tells them apart by
    /// handing the value to that Python type. This crate has no such type at parse time, so a
    /// structured field is left for the caller's own typing to judge rather than guessed at.
    pub(crate) fn coerce_scalars(&self, value: &mut Value) -> Result<()> {
        for field in &self.outputs {
            if matches!(field.kind, FieldKind::Json(_)) {
                continue;
            }
            if let Some(entry) = value.get_mut(&field.name) {
                coerce_value(&field.kind, &field.name, entry)?;
            }
        }
        Ok(())
    }

    pub(crate) fn coerce(&self, value: &mut Value) -> Result<()> {
        for field in &self.outputs {
            if let Some(entry) = value.get_mut(&field.name) {
                coerce_value(&field.kind, &field.name, entry)?;
            }
        }
        Ok(())
    }

    /// Every declared field must come back, and closed sets must hold. The error doubles as
    /// retry feedback the model reads, so it states the requirement. Typed fields arrive
    /// here already coerced, so only presence is left to check. Task-specific checks
    /// (ranges, scrubbing) stay with the task's own parser.
    pub(crate) fn ensure(&self, value: &Value) -> Result<()> {
        for field in &self.outputs {
            let missing = || anyhow!("the {} field is missing and is required", field.name);
            if field.kind != FieldKind::Str {
                value.get(&field.name).ok_or_else(missing)?;
                continue;
            }
            let got = value
                .get(&field.name)
                .and_then(Value::as_str)
                .ok_or_else(missing)?;
            if let Some(values) = &field.values
                && !values.iter().any(|value| value.wire_form() == got)
            {
                return Err(anyhow!(
                    "{} must be one of {}; got {got:?}",
                    field.name,
                    wire_forms(values, ", ")
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature() -> super::Signature {
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

    fn typed_out(name: &str, kind: FieldKind) -> OutField {
        OutField {
            name: name.into(),
            desc: name.into(),
            kind,
            ..Default::default()
        }
    }

    fn typed_signature() -> super::Signature {
        Signature::single_input(
            "Size the gift.",
            vec![
                typed_out("note", FieldKind::Str),
                typed_out("double", FieldKind::Bool),
                typed_out("count", FieldKind::Int),
                typed_out("amount", FieldKind::Float),
            ],
        )
    }

    #[test]
    fn a_closed_set_is_the_fields_annotation_and_a_bare_kind_is_not() {
        let sig = signature();
        assert_eq!(sig.outputs[0].annotation(), "Literal['red', 'blue']");
        assert_eq!(sig.outputs[1].annotation(), "str");
    }

    #[test]
    fn single_input_declares_the_request_field() {
        let sig = signature();
        assert_eq!(sig.inputs.len(), 1);
        assert_eq!(sig.inputs[0].name, "request");
        assert_eq!(sig.inputs[0].desc, "the request");
    }

    #[test]
    fn schema_lists_every_field_and_closed_set() {
        let schema = signature().schema();
        assert_eq!(schema["required"], json!(["color", "why"]));
        assert_eq!(
            schema["properties"]["color"]["enum"],
            json!(["red", "blue"])
        );
        assert_eq!(schema["properties"]["why"]["enum"], Value::Null);
    }

    #[test]
    fn output_clause_names_keys_descriptions_and_values() {
        let clause = signature().output_clause();
        assert!(clause.contains("keys color, why"));
        assert!(clause.contains("color: the chosen color (one of: red, blue)"));
        assert!(clause.contains("why: one short sentence"));
    }

    #[test]
    fn ensure_rejects_missing_fields_and_out_of_set_values() {
        let sig = signature();
        assert!(
            sig.ensure(&json!({ "color": "red", "why": "calm" }))
                .is_ok()
        );
        assert!(sig.ensure(&json!({ "color": "red" })).is_err());
        assert!(
            sig.ensure(&json!({ "color": "green", "why": "x" }))
                .is_err()
        );
    }

    #[test]
    fn schema_types_follow_field_kinds() {
        let schema = typed_signature().schema();
        assert_eq!(schema["properties"]["note"]["type"], json!("string"));
        assert_eq!(schema["properties"]["double"]["type"], json!("boolean"));
        assert_eq!(schema["properties"]["count"]["type"], json!("integer"));
        assert_eq!(schema["properties"]["amount"]["type"], json!("number"));
    }

    #[test]
    fn output_clause_annotates_typed_fields_and_leaves_str_alone() {
        let clause = typed_signature().output_clause();
        assert!(clause.contains("note: note."));
        assert!(clause.contains("double (bool): double."));
        assert!(clause.contains("count (int): count."));
        assert!(clause.contains("amount (float): amount."));
    }

    #[test]
    fn coerce_parses_marker_strings_into_native_values() {
        let sig = typed_signature();
        let mut value = json!({
            "note": "hi",
            "double": " true ",
            "count": "-7",
            "amount": " 0.04 ",
        });
        sig.coerce(&mut value).expect("coerces");
        assert_eq!(
            value,
            json!({ "note": "hi", "double": true, "count": -7, "amount": 0.04 })
        );
        assert!(sig.ensure(&value).is_ok());
    }

    #[test]
    fn coerce_reads_a_bool_back_in_pythons_spelling() {
        // The prompt asks for `True`/`False` and a demo renders one that way, so the parser
        // has to accept the spelling the model was shown.
        for (text, expected) in [("True", true), ("False", false), ("TRUE", true)] {
            let sig = typed_signature();
            let mut value = json!({ "note": "hi", "double": text, "count": 1, "amount": 0.5 });
            sig.coerce(&mut value).expect("coerces");
            assert_eq!(value["double"], json!(expected));
        }
    }

    #[test]
    fn coerce_accepts_native_json_values_as_is() {
        let sig = typed_signature();
        let mut value = json!({ "note": "hi", "double": false, "count": 3, "amount": 2 });
        sig.coerce(&mut value).expect("coerces");
        // An integer is a number; models in JSON mode often drop the decimal point.
        assert_eq!(
            value,
            json!({ "note": "hi", "double": false, "count": 3, "amount": 2 })
        );
    }

    #[test]
    fn coerce_rejects_malformed_typed_values_with_feedback_errors() {
        let sig = typed_signature();
        for (patch, message) in [
            (
                json!({ "double": "maybe" }),
                "double must be true or false, got \"maybe\"",
            ),
            (
                json!({ "double": 1 }),
                "double must be true or false, got 1",
            ),
            (
                json!({ "count": "3.5" }),
                "count must be an integer, got \"3.5\"",
            ),
            (json!({ "count": 3.5 }), "count must be an integer, got 3.5"),
            (
                json!({ "amount": "abc" }),
                "amount must be a number, got \"abc\"",
            ),
            (
                json!({ "amount": "inf" }),
                "amount must be a number, got \"inf\"",
            ),
            (
                json!({ "amount": "NaN" }),
                "amount must be a number, got \"NaN\"",
            ),
        ] {
            let mut value = json!({ "note": "hi", "double": true, "count": 1, "amount": 0.01 });
            for (key, bad) in patch.as_object().expect("object") {
                value[key] = bad.clone();
            }
            let error = sig.coerce(&mut value).expect_err("rejects").to_string();
            assert_eq!(error, message);
        }
    }

    #[test]
    fn coerce_skips_missing_fields_and_ensure_reports_them() {
        let sig = typed_signature();
        let mut value = json!({ "note": "hi", "double": true, "count": 1 });
        sig.coerce(&mut value)
            .expect("missing is not a coercion error");
        let error = sig.ensure(&value).expect_err("missing").to_string();
        assert_eq!(error, "the amount field is missing and is required");
    }

    fn ideas_schema() -> Value {
        json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": { "title": { "type": "string" } },
                "required": ["title"],
            },
        })
    }

    fn json_signature() -> super::Signature {
        Signature::single_input(
            "Suggest ideas.",
            vec![OutField {
                name: "ideas".into(),
                desc: "three concrete ideas".into(),
                kind: FieldKind::opaque_json(),
                schema: Some(ideas_schema()),
                ..Default::default()
            }],
        )
    }

    #[test]
    fn schema_embeds_the_nested_schema_of_a_json_field() {
        let schema = json_signature().schema();
        assert_eq!(schema["properties"]["ideas"], ideas_schema());
        assert_eq!(schema["required"], json!(["ideas"]));
    }

    #[test]
    fn a_json_field_without_a_schema_accepts_any_value() {
        let mut sig = json_signature();
        sig.outputs[0].schema = None;
        assert_eq!(sig.schema()["properties"]["ideas"], json!({}));
        let clause = sig.output_clause();
        assert!(clause.contains("ideas (json): three concrete ideas."));
    }

    #[test]
    fn output_clause_carries_the_json_schema_in_one_line() {
        let clause = json_signature().output_clause();
        let expected = format!(
            "ideas: three concrete ideas (json matching schema: {}).",
            ideas_schema()
        );
        assert!(clause.contains(&expected), "got: {clause}");
        assert!(!clause.contains('\n'));
    }

    #[test]
    fn coerce_passes_native_json_values_and_parses_string_forms() {
        let sig = json_signature();
        for (reply, parsed) in [
            (
                json!({ "ideas": [{ "title": "a" }] }),
                json!([{ "title": "a" }]),
            ),
            (
                json!({ "ideas": r#" [{"title": "a"}] "# }),
                json!([{ "title": "a" }]),
            ),
            (
                json!({ "ideas": "```json\n[{\"title\": \"a\"}]\n```" }),
                json!([{ "title": "a" }]),
            ),
            (
                json!({ "ideas": "```\n{\"title\": \"a\"}\n```" }),
                json!({ "title": "a" }),
            ),
        ] {
            let mut value = reply;
            sig.coerce(&mut value).expect("coerces");
            assert_eq!(value["ideas"], parsed);
            assert!(sig.ensure(&value).is_ok());
        }
    }

    #[test]
    fn coerce_rejects_a_json_field_that_does_not_parse() {
        let sig = json_signature();
        let mut value = json!({ "ideas": "three great ideas" });
        let error = sig.coerce(&mut value).expect_err("rejects").to_string();
        assert!(
            error.starts_with("ideas must be valid JSON: "),
            "got: {error}"
        );
    }

    #[test]
    fn json_field_schema_inlines_subschemas_and_drops_root_metadata() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Idea {
            title: String,
            why: String,
        }
        let schema = json_field_schema::<Vec<Idea>>();
        assert_eq!(schema["type"], json!("array"));
        assert_eq!(
            schema["items"]["properties"]["why"]["type"],
            json!("string")
        );
        assert_eq!(schema["items"]["required"], json!(["title", "why"]));
        let rendered = schema.to_string();
        assert!(!rendered.contains("$ref"), "got: {rendered}");
        assert!(!rendered.contains("$schema"), "got: {rendered}");
        assert_eq!(schema.get("title"), None);
    }

    /// The derive is declaration data; the structs themselves are never built.
    #[allow(dead_code)]
    mod derived {
        use crate::signature::Signature;

        /// Grade the essay.
        #[derive(Signature)]
        pub struct DocTask {
            #[input]
            pub essay: String,
            #[output]
            pub grade: String,
        }

        /// Ignored in favor of the attribute.
        #[derive(Signature)]
        #[signature(instructions = "Attribute instructions win.")]
        pub struct AttrTask {
            /// the topic to rhyme on
            #[input]
            pub topic: String,
            #[input(desc = "the mood to strike")]
            /// ignored in favor of the attribute
            pub mood: String,
            #[output(desc = "two rhyming lines")]
            pub couplet: String,
            #[output(values("playful", "solemn"))]
            pub tone: String,
        }

        /// Judge the pitch.
        #[derive(Signature)]
        pub struct JudgeTask {
            #[input]
            pub pitch: String,
            #[input(desc = "the asking price")]
            pub price: f64,
            #[input]
            pub urgent: bool,
            #[output(desc = "fund it or not")]
            pub fund: bool,
            #[output(desc = "the counter offer")]
            pub counter: f32,
            #[output(desc = "negotiation rounds")]
            pub rounds: u8,
        }
    }

    use derived::{AttrTask, DocTask};

    #[test]
    fn derive_takes_instructions_from_the_doc_comment() {
        assert_eq!(DocTask::signature().instructions, "Grade the essay.");
    }

    #[test]
    fn derive_prefers_attribute_instructions_over_the_doc_comment() {
        assert_eq!(
            AttrTask::signature().instructions,
            "Attribute instructions win."
        );
    }

    /// A description comes from the attribute, then the doc comment, and then nowhere.
    ///
    /// Not from the field's own name: dspy stores `${name}` for an undescribed field and drops it
    /// again when rendering, so the name never reaches a prompt. Substituting it here put it on
    /// every field line that carried no `desc` — visible only by diffing a whole message against
    /// dspy's, since no fixture exercises the derive.
    #[test]
    fn derive_desc_falls_back_from_attribute_to_doc_to_nothing() {
        let sig = AttrTask::signature();
        assert_eq!(sig.inputs[0].desc, "the topic to rhyme on");
        assert_eq!(sig.inputs[1].desc, "the mood to strike");
        assert_eq!(sig.outputs[0].desc, "two rhyming lines");
        assert_eq!(sig.outputs[1].desc, "", "an undescribed field says nothing");

        // Neither an attribute nor a doc comment, so nothing — where this used to answer
        // "essay" and "grade".
        let doc = DocTask::signature();
        assert_eq!(doc.inputs[0].desc, "");
        assert_eq!(doc.outputs[0].desc, "");
    }

    #[test]
    fn derive_carries_closed_sets_on_outputs() {
        let sig = AttrTask::signature();
        assert_eq!(sig.outputs[0].values, None);
        assert_eq!(
            sig.outputs[1].values,
            Some(vec!["playful".into(), "solemn".into()])
        );
    }

    #[test]
    fn derive_keeps_declaration_order_and_renders_pairs_from_it() {
        let sig = AttrTask::signature();
        let names: Vec<&str> = sig.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["topic", "mood"]);
        let outs: Vec<&str> = sig.outputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(outs, ["couplet", "tone"]);

        let pairs = AttrTask::input_pairs(&derived::AttrTaskInputs {
            topic: "rain".into(),
            mood: "wistful".into(),
        });
        assert_eq!(
            pairs,
            vec![
                crate::adapter::Input::new("topic", json!("rain")),
                crate::adapter::Input::new("mood", json!("wistful")),
            ],
            "two String fields are loose values, not records"
        );
    }

    #[test]
    fn derived_outputs_deserialize_from_the_validated_reply() {
        let outputs: derived::AttrTaskOutputs =
            serde_json::from_value(json!({ "couplet": "a\nb", "tone": "playful" }))
                .expect("deserializes");
        assert_eq!(outputs.couplet, "a\nb");
        assert_eq!(outputs.tone, "playful");
    }

    #[test]
    fn derive_maps_rust_types_to_field_kinds() {
        let sig = derived::JudgeTask::signature();
        let in_kinds: Vec<FieldKind> = sig.inputs.iter().map(|f| f.kind.clone()).collect();
        assert_eq!(
            in_kinds,
            [FieldKind::Str, FieldKind::Float, FieldKind::Bool]
        );
        let out_kinds: Vec<FieldKind> = sig.outputs.iter().map(|f| f.kind.clone()).collect();
        assert_eq!(
            out_kinds,
            [FieldKind::Bool, FieldKind::Float, FieldKind::Int]
        );
        assert_eq!(
            sig.schema()["properties"]["rounds"]["type"],
            json!("integer")
        );
    }

    #[test]
    fn derive_renders_typed_inputs_through_to_string() {
        let pairs = derived::JudgeTask::input_pairs(&derived::JudgeTaskInputs {
            pitch: "a bakery".into(),
            price: 0.04,
            urgent: true,
        });
        // A scalar keeps its type across the boundary, which is what lets the adapter reach
        // Python's spelling: dspy 3.2.1 renders this `urgent` as `True`, not `true`.
        assert_eq!(
            pairs,
            vec![
                crate::adapter::Input::new("pitch", json!("a bakery")),
                crate::adapter::Input::new("price", json!(0.04)),
                crate::adapter::Input::new("urgent", json!(true)),
            ]
        );
    }

    #[test]
    fn derived_typed_outputs_deserialize_from_native_json() {
        let outputs: derived::JudgeTaskOutputs =
            serde_json::from_value(json!({ "fund": true, "counter": 0.02, "rounds": 3 }))
                .expect("deserializes");
        assert!(outputs.fund);
        assert_eq!(outputs.counter, 0.02);
        assert_eq!(outputs.rounds, 3);
    }
}
