//! What a model request becomes when the model is a coding agent: one prompt,
//! standing instructions, and any images — because an agent CLI takes a
//! prompt, not a conversation.

use base64::Engine;
use dsrust::lm::api::{LmMessage, LmPart, LmRequest, LmSource};
use harness::Attachment;

/// A request rendered for an agent run.
pub(crate) struct Rendered {
    /// What the agent is asked. A single user message goes through verbatim —
    /// an adapter's field markers must arrive exactly as rendered — and a
    /// longer exchange becomes a labelled transcript ending on the latest turn.
    pub prompt: String,
    /// System and developer messages, in order, for the harness's
    /// `extra_instructions` — appended to the agent's own system prompt.
    pub instructions: Option<String>,
    /// Images the request carried as data or as a local file. An image only
    /// reachable by URL is named in the prompt instead: a run has no client to
    /// fetch it with, and the adapters that take images take bytes.
    pub attachments: Vec<Attachment>,
}

pub(crate) fn render(request: &LmRequest) -> Rendered {
    let (system, turns): (Vec<&LmMessage>, Vec<&LmMessage>) = request
        .messages
        .iter()
        .partition(|m| matches!(m.role.as_str(), "system" | "developer"));
    let instructions = join(system.iter().filter_map(|m| m.text()));

    let mut attachments = Vec::new();
    let mut lines = Vec::new();
    for message in &turns {
        let text = text_with_images(message, &mut attachments);
        lines.push(match turns.len() {
            1 => text,
            _ => format!("[{}]\n{text}", message.role),
        });
    }
    Rendered {
        prompt: lines.join("\n\n"),
        instructions,
        attachments,
    }
}

/// A message's text, with each image either lifted out as an attachment or,
/// when it is only a URL, named in place.
fn text_with_images(message: &LmMessage, attachments: &mut Vec<Attachment>) -> String {
    let mut text = String::new();
    for part in &message.parts {
        match part {
            LmPart::Text { text: t, .. } => text.push_str(t),
            LmPart::Image {
                source, media_type, ..
            } => match bytes(source) {
                Some(data) => attachments.push(Attachment {
                    mime_type: media_type.clone(),
                    data,
                }),
                None => text.push_str(&format!("\n(image: {})", describe(source))),
            },
            _ => {}
        }
    }
    text
}

fn bytes(source: &LmSource) -> Option<Vec<u8>> {
    match source {
        LmSource::Data(encoded) => {
            // A data URI carries its own header; a bare payload does not.
            let payload = encoded
                .rsplit_once(",")
                .map_or(encoded.as_str(), |(_, p)| p);
            base64::engine::general_purpose::STANDARD
                .decode(payload)
                .ok()
        }
        LmSource::Path(path) => std::fs::read(path).ok(),
        LmSource::Url(_) | LmSource::FileId(_) => None,
    }
}

fn describe(source: &LmSource) -> String {
    match source {
        LmSource::Url(url) => url.clone(),
        LmSource::FileId(id) => format!("file {id}"),
        LmSource::Path(path) => path.display().to_string(),
        LmSource::Data(_) => "inline data".to_owned(),
    }
}

fn join(texts: impl Iterator<Item = String>) -> Option<String> {
    let joined = texts
        .filter(|t| !t.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!joined.is_empty()).then_some(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<LmMessage>) -> LmRequest {
        LmRequest::from_messages("m", messages)
    }

    #[test]
    fn one_user_message_is_the_prompt_verbatim_and_system_becomes_instructions() {
        let r = render(&request(vec![
            LmMessage::system(["be terse"]),
            LmMessage::user(["[[ ## question ## ]]\nwhy?\n\n[[ ## answer ## ]]"]),
        ]));
        assert_eq!(
            r.prompt, "[[ ## question ## ]]\nwhy?\n\n[[ ## answer ## ]]",
            "markers untouched"
        );
        assert_eq!(r.instructions.as_deref(), Some("be terse"));
        assert!(r.attachments.is_empty());
    }

    #[test]
    fn a_conversation_becomes_a_labelled_transcript_ending_on_the_latest_turn() {
        let r = render(&request(vec![
            LmMessage::user(["hi"]),
            LmMessage::assistant(["hello"]),
            LmMessage::user(["and?"]),
        ]));
        assert_eq!(r.prompt, "[user]\nhi\n\n[assistant]\nhello\n\n[user]\nand?");
        assert_eq!(r.instructions, None);
    }

    #[test]
    fn developer_messages_join_the_instructions_and_blank_ones_are_dropped() {
        let r = render(&request(vec![
            LmMessage::system(["one"]),
            LmMessage::developer(["   "]),
            LmMessage::developer(["two"]),
            LmMessage::user(["q"]),
        ]));
        assert_eq!(r.instructions.as_deref(), Some("one\n\ntwo"));
    }

    #[test]
    fn inline_image_data_becomes_an_attachment_and_a_url_is_named_in_the_prompt() {
        let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG");
        let r = render(&request(vec![LmMessage::user([
            LmPart::text("look"),
            LmPart::Image {
                source: LmSource::Data(format!("data:image/png;base64,{png}")),
                media_type: "image/png".into(),
                detail: None,
                metadata: Default::default(),
            },
            LmPart::image_url("https://example.test/a.png"),
        ])]));
        assert_eq!(r.attachments.len(), 1);
        assert_eq!(r.attachments[0].data, b"\x89PNG");
        assert_eq!(r.attachments[0].mime_type, "image/png");
        assert!(
            r.prompt.contains("(image: https://example.test/a.png)"),
            "{}",
            r.prompt
        );
    }
}
