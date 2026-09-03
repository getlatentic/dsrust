//! The markdown a multimodal reflective dataset becomes, and the media pulled out of it.
//!
//! Not gepa's renderer. dspy's multimodal proposer carries its own, and the two differ in every
//! detail that shows: headings start at level 3 inside a section and are capped at 6, an empty map
//! or list still emits a blank line, and a custom type is replaced by a numbered placeholder whose
//! count restarts for each example.

use gepa::Reflective;

use crate::adapter::types::base::{CUSTOM_TYPE_END, CUSTOM_TYPE_START};
use crate::optimize::gepa::proposer::ReflectiveDataset;

/// dspy's deepest heading. `min(level + 1, 6)` in the recursion, so a structure nested further
/// keeps writing `######`.
const DEEPEST: usize = 6;

/// The examples as markdown, and every custom type they held, in the order they were met.
///
/// The header line is prepended only when at least one was found, and counts them across the whole
/// dataset — while the placeholders inside each example are numbered from one again.
pub(super) fn formatted(dataset: &ReflectiveDataset) -> (String, Vec<String>) {
    let mut media = Vec::new();
    let mut parts = Vec::with_capacity(dataset.len());
    for (index, sample) in dataset.iter().enumerate() {
        // Upstream builds `example_images` inside `convert_sample_to_markdown_with_images`, so
        // `[IMAGE-1]` in the second example is that example's first and not the dataset's.
        let mut here = Vec::new();
        let mut text = format!("# Example {}\n", index + 1);
        for (key, value) in sample {
            text.push_str(&format!("## {key}\n"));
            text.push_str(&render(value, 3, &mut here));
        }
        parts.push(text);
        media.extend(here);
    }
    let mut text = parts.join("\n\n");
    if !media.is_empty() {
        text = format!(
            "The examples below include visual content ({} images total). Please analyze both the \
             text and visual elements when suggesting improvements.\n\n{text}",
            media.len()
        );
    }
    (text, media)
}

fn render(value: &Reflective, level: usize, here: &mut Vec<String>) -> String {
    match value {
        // Upstream asks `isinstance(value, Type)`. A custom type reaches a field here as its
        // serialized form — the sentinels around a JSON block list — which is the same text
        // Python's `str()` gives it, so the question is asked of the value rather than of a class.
        Reflective::Text(text) if is_custom_type(text) => {
            here.push(text.clone());
            format!("[IMAGE-{} - see visual content]\n\n", here.len())
        }
        Reflective::Text(text) => format!("{}\n\n", text.trim()),
        Reflective::Map(entries) => {
            let mut out = String::new();
            for (key, value) in entries {
                out.push_str(&format!("{} {key}\n", "#".repeat(level)));
                out.push_str(&render(value, (level + 1).min(DEEPEST), here));
            }
            // An empty map still writes a blank line, so the section is not silently absent.
            if entries.is_empty() {
                out.push('\n');
            }
            out
        }
        Reflective::List(items) => {
            let mut out = String::new();
            for (index, item) in items.iter().enumerate() {
                out.push_str(&format!("{} Item {}\n", "#".repeat(level), index + 1));
                out.push_str(&render(item, (level + 1).min(DEEPEST), here));
            }
            if items.is_empty() {
                out.push('\n');
            }
            out
        }
    }
}

fn is_custom_type(text: &str) -> bool {
    text.starts_with(CUSTOM_TYPE_START) && text.ends_with(CUSTOM_TYPE_END)
}
