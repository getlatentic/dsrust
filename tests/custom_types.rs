//! A Rust custom type in a signature field renders the way dspy's does.
//!
//! dspy keeps a live `Type` object and serializes it into the field; here the value is the type's
//! serialized form (the sentinel-wrapped blocks, or a bare string for a text-like type), and the
//! adapter's `split_custom_types` reads the blocks back into a multimodal message. These exercise
//! that whole path from a Rust value — the reach that previously existed only for a pre-serialized
//! string crossing from the Python bridge.

use dsrust::adapter::{Adapter, ChatAdapter, Input};
use dsrust::lm::Content;
use dsrust::signature::{FieldKind, InField, JsonType, OutField, Signature, SignatureSpec};
use dsrust::{Audio, Code, File, Image};
use serde_json::json;

/// `question` plus a custom-type input, answering `answer`.
fn signature_with(custom_field: &str, annotation: &str) -> Signature {
    Signature {
        instructions: "Answer.".into(),
        inputs: vec![
            InField { name: "question".into(), ..Default::default() },
            InField {
                name: custom_field.into(),
                kind: FieldKind::Json(JsonType::plain(annotation)),
                ..Default::default()
            },
        ],
        outputs: vec![OutField { name: "answer".into(), ..Default::default() }],
    }
}

/// The user turn a `ChatAdapter` renders for these inputs.
fn user_content(signature: &Signature, inputs: &[Input<'_>]) -> Content {
    let (_system, turns) = ChatAdapter::default().format(signature, &[], inputs).expect("formats");
    turns.into_iter().next_back().expect("a user turn").content
}

/// An `Image` value becomes an `image_url` content block, split out of the field text by the
/// sentinels — the multimodal path, now reachable from a Rust value.
#[test]
fn an_image_field_becomes_an_image_url_block() {
    let signature = signature_with("photo", "Image");
    let inputs = [
        Input::new("question", json!("describe")),
        Input::new("photo", serde_json::to_value(Image::new("https://example.com/a.jpg")).unwrap()),
    ];
    let Content::Blocks(blocks) = user_content(&signature, &inputs) else {
        panic!("an image field renders a multimodal message");
    };
    assert!(blocks.contains(
        &json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.jpg" } })
    ));
}

/// An `Audio` value becomes an `input_audio` block.
#[test]
fn an_audio_field_becomes_an_input_audio_block() {
    let signature = signature_with("clip", "Audio");
    let inputs = [
        Input::new("question", json!("transcribe")),
        Input::new("clip", serde_json::to_value(Audio::new("QUJD", "wav")).unwrap()),
    ];
    let Content::Blocks(blocks) = user_content(&signature, &inputs) else {
        panic!("an audio field renders a multimodal message");
    };
    assert!(blocks.contains(
        &json!({ "type": "input_audio", "input_audio": { "data": "QUJD", "format": "wav" } })
    ));
}

/// A `File` value becomes a `file` block carrying only the fields that were set.
#[test]
fn a_file_field_becomes_a_file_block() {
    let signature = signature_with("doc", "File");
    let inputs = [
        Input::new("question", json!("summarize")),
        Input::new("doc", serde_json::to_value(File::from_id("file-1").with_filename("a.txt")).unwrap()),
    ];
    let Content::Blocks(blocks) = user_content(&signature, &inputs) else {
        panic!("a file field renders a multimodal message");
    };
    assert!(blocks.contains(
        &json!({ "type": "file", "file": { "file_id": "file-1", "filename": "a.txt" } })
    ));
}

/// A `Code` value renders as text — it bypasses the sentinels, so the message stays a string and
/// carries the code inline, the way dspy's `Code.serialize_model` override does.
#[test]
fn a_code_field_renders_inline_as_text() {
    let signature = signature_with("snippet", "Code");
    let inputs = [
        Input::new("question", json!("analyze")),
        Input::new("snippet", serde_json::to_value(Code::new("x = 1")).unwrap()),
    ];
    let content = user_content(&signature, &inputs);
    let text = content.text().expect("code renders as text, not a block");
    assert!(text.contains("[[ ## snippet ## ]]\nx = 1"), "got: {text}");
}

/// A custom type in a `#[derive(Signature)]` input field: dspy's `code: dspy.Code = InputField()`.
/// The derive maps it to a field named for the type, and a value of it renders through that field —
/// the ergonomic path, no hand-built signature.
#[allow(dead_code)]
#[derive(dsrust::Signature)]
/// Analyze the code.
struct Analyze {
    #[input]
    question: String,
    #[input]
    code: Code,
    #[input]
    photo: Image,
    #[output]
    answer: String,
}

#[test]
fn a_derived_signature_maps_custom_type_input_fields() {
    let signature = Analyze::signature();
    let inputs: Vec<&str> = signature.inputs.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(inputs, ["question", "code", "photo"]);
    // The derive names each field by its type — the annotation the whole adapter branches on.
    assert_eq!(signature.inputs[1].annotation(), "Code");
    assert_eq!(signature.inputs[2].annotation(), "Image");

    // And values of those types render through the derived fields: the image as a block, the code
    // inline.
    let rendered = ChatAdapter::default()
        .format(
            &signature,
            &[],
            &[
                Input::new("question", json!("what is this")),
                Input::new("code", serde_json::to_value(Code::new("x = 1")).unwrap()),
                Input::new("photo", serde_json::to_value(Image::new("https://x/a.jpg")).unwrap()),
            ],
        )
        .expect("renders");
    let Content::Blocks(blocks) = &rendered.1.last().expect("a user turn").content else {
        panic!("a derived image input renders a multimodal message");
    };
    assert!(blocks.iter().any(|block| block.get("type").and_then(|t| t.as_str()) == Some("image_url")));
    let text: String = blocks.iter().filter_map(|b| b.get("text").and_then(|t| t.as_str())).collect();
    assert!(text.contains("[[ ## code ## ]]\nx = 1"), "the code renders inline: {text}");
}
