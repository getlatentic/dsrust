//! dspy `adapters/types/citation.py`: the `Citations` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

use crate::signature::TypeDescription;

use super::base::{Formatted, Type, serialized};

/// dspy's `Citations.Citation`: one quoted span and where it came from.
///
/// A provider that supports citations — Anthropic's through litellm — returns these beside the
/// answer, each naming the document it quoted and the character range within it.
///
/// ```
/// use dsrust::Citation;
///
/// // `document_index` points into the documents the request carried, so a citation is resolved
/// // against what was sent rather than carrying the source itself.
/// let cited = Citation {
///     kind: "char_location".to_owned(),
///     cited_text: "Paris is the capital.".to_owned(),
///     document_index: 0,
///     document_title: Some("France".to_owned()),
///     start_char_index: 0,
///     end_char_index: 21,
///     supported_text: None,
/// };
/// assert_eq!(cited.document_index, 0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
pub struct Citation {
    /// dspy's `type`, which only ever carries `char_location` today. Spelled `kind` because
    /// `type` is a Rust keyword; the wire name is unchanged.
    #[serde(rename = "type", default = "char_location")]
    pub kind: String,
    pub cited_text: String,
    pub document_index: i64,
    #[serde(default)]
    pub document_title: Option<String>,
    pub start_char_index: i64,
    pub end_char_index: i64,
    #[serde(default)]
    pub supported_text: Option<String>,
}

fn char_location() -> String {
    "char_location".to_owned()
}

impl Citation {
    /// A citation of a character range, with no title or supported text.
    pub fn new(cited_text: impl Into<String>, document_index: i64, range: (i64, i64)) -> Self {
        Self {
            kind: char_location(),
            cited_text: cited_text.into(),
            document_index,
            document_title: None,
            start_char_index: range.0,
            end_char_index: range.1,
            supported_text: None,
        }
    }

    pub fn document_title(mut self, title: impl Into<String>) -> Self {
        self.document_title = Some(title.into());
        self
    }

    pub fn supported_text(mut self, supported: impl Into<String>) -> Self {
        self.supported_text = Some(supported.into());
        self
    }

    /// dspy `Citation.format`: the required keys first, then the two optional ones, each omitted
    /// where it is empty — the order upstream builds the mapping in.
    pub fn format(&self) -> Value {
        let mut citation = Map::new();
        citation.insert("type".to_owned(), json!(self.kind));
        citation.insert("cited_text".to_owned(), json!(self.cited_text));
        citation.insert("document_index".to_owned(), json!(self.document_index));
        citation.insert("start_char_index".to_owned(), json!(self.start_char_index));
        citation.insert("end_char_index".to_owned(), json!(self.end_char_index));
        // dspy tests each for Python truth, so an empty string is dropped as well as a missing one.
        if let Some(title) = self
            .document_title
            .as_deref()
            .filter(|title| !title.is_empty())
        {
            citation.insert("document_title".to_owned(), json!(title));
        }
        if let Some(supported) = self
            .supported_text
            .as_deref()
            .filter(|text| !text.is_empty())
        {
            citation.insert("supported_text".to_owned(), json!(supported));
        }
        Value::Object(citation)
    }
}

/// dspy's `Citations`: the citations an answer rests on, as a field of its own.
///
/// Ordinarily an output field — a provider fills it beside the answer, and
/// [`parse_lm_response`](Type::parse_lm_response) is what reads it back off the reply.
#[derive(Debug, Clone, PartialEq, Eq, Default, schemars::JsonSchema)]
pub struct Citations {
    pub citations: Vec<Citation>,
}

impl Citations {
    pub fn new(citations: impl IntoIterator<Item = Citation>) -> Self {
        Self {
            citations: citations.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.citations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.citations.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Citation> {
        self.citations.iter()
    }
}

impl std::ops::Index<usize> for Citations {
    type Output = Citation;

    fn index(&self, index: usize) -> &Citation {
        &self.citations[index]
    }
}

impl<'a> IntoIterator for &'a Citations {
    type Item = &'a Citation;
    type IntoIter = std::slice::Iter<'a, Citation>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Type for Citations {
    /// dspy `Citations.format`: each citation as its own mapping.
    fn format(&self) -> Formatted {
        Formatted::Blocks(self.citations.iter().map(Citation::format).collect())
    }

    fn description() -> Option<TypeDescription> {
        Some(TypeDescription {
            name: "Citations".to_owned(),
            text: "Citations with quoted text and source references. Include the exact text \
                   being cited and information about its source."
                .to_owned(),
            replaces_schema: false,
        })
    }

    /// dspy `Citations.parse_lm_response`: a reply carrying a `citations` list becomes this field's
    /// value directly, without going through the text the adapter would otherwise parse.
    fn parse_lm_response(response: &Value) -> Option<Self> {
        let citations = response.get("citations")?.as_array()?;
        citations
            .iter()
            .map(|citation| serde_json::from_value(citation.clone()).ok())
            .collect::<Option<Vec<Citation>>>()
            .map(Self::new)
    }

    /// dspy streams citations as they arrive, one `citation` field per chunk.
    fn is_streamable() -> bool {
        true
    }
}

impl Serialize for Citations {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl<'de> Deserialize<'de> for Citations {
    /// dspy `Citations.validate_input`: a list of citation mappings, a mapping carrying one under
    /// `citations`, or a single citation mapping — anything else is refused.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let invalid = || {
            de::Error::custom(format!(
                "Received invalid value for `Citations`: {}",
                crate::python::text(&value)
            ))
        };
        let citations = match &value {
            Value::Array(items) if items.iter().all(has_cited_text) => items.clone(),
            Value::Object(fields) => match fields.get("citations") {
                Some(Value::Array(items)) => items.clone(),
                Some(_) => return Err(invalid()),
                None if fields.contains_key("cited_text") => vec![value.clone()],
                None => return Err(invalid()),
            },
            _ => return Err(invalid()),
        };
        citations
            .into_iter()
            .map(|citation| serde_json::from_value(citation).map_err(de::Error::custom))
            .collect::<Result<Vec<Citation>, _>>()
            .map(Self::new)
    }
}

fn has_cited_text(item: &Value) -> bool {
    item.as_object()
        .is_some_and(|fields| fields.contains_key("cited_text"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::base::{CUSTOM_TYPE_END, CUSTOM_TYPE_START};

    fn quoted() -> Citation {
        Citation::new("The sky is blue", 0, (0, 15))
    }

    /// The required keys come first in upstream's order, and each optional one appears only when
    /// it carries something.
    #[test]
    fn a_citation_formats_its_required_keys_then_the_ones_it_has() {
        assert_eq!(
            serde_json::to_string(&quoted().format()).expect("serializes"),
            r#"{"type":"char_location","cited_text":"The sky is blue","document_index":0,"start_char_index":0,"end_char_index":15}"#
        );
        let full = quoted()
            .document_title("Weather")
            .supported_text("It was blue.");
        assert_eq!(
            serde_json::to_string(&full.format()).expect("serializes"),
            r#"{"type":"char_location","cited_text":"The sky is blue","document_index":0,"start_char_index":0,"end_char_index":15,"document_title":"Weather","supported_text":"It was blue."}"#
        );
    }

    #[test]
    fn it_renders_as_sentinel_wrapped_citation_blocks() {
        let citations = Citations::new([quoted()]);
        assert_eq!(
            citations.format(),
            Formatted::Blocks(vec![quoted().format()])
        );
        let rendered = serde_json::to_value(&citations).expect("serializes");
        let rendered = rendered.as_str().expect("a string");
        assert!(rendered.starts_with(CUSTOM_TYPE_START) && rendered.ends_with(CUSTOM_TYPE_END));
    }

    /// dspy's validator takes a list, a `citations` mapping, or a lone citation; anything else is
    /// an error.
    #[test]
    fn it_reads_each_shape_dspy_accepts() {
        let one = json!({
            "cited_text": "The sky is blue",
            "document_index": 0,
            "start_char_index": 0,
            "end_char_index": 15,
        });
        let expected = Citations::new([quoted()]);
        assert_eq!(
            serde_json::from_value::<Citations>(json!([one.clone()])).expect("a list"),
            expected
        );
        assert_eq!(
            serde_json::from_value::<Citations>(json!({ "citations": [one.clone()] }))
                .expect("a mapping"),
            expected
        );
        assert_eq!(
            serde_json::from_value::<Citations>(one).expect("a lone citation"),
            expected
        );
        assert!(serde_json::from_value::<Citations>(json!("nope")).is_err());
        assert!(serde_json::from_value::<Citations>(json!({ "other": 1 })).is_err());
    }

    /// A reply carrying citations fills the field directly; one without leaves it alone.
    #[test]
    fn it_parses_citations_off_a_reply() {
        let response = json!({
            "citations": [{
                "type": "char_location",
                "cited_text": "The sky is blue",
                "document_index": 0,
                "start_char_index": 0,
                "end_char_index": 15,
            }],
        });
        assert_eq!(
            Citations::parse_lm_response(&response).expect("citations"),
            Citations::new([quoted()])
        );
        assert!(Citations::parse_lm_response(&json!({ "answer": "blue" })).is_none());
        assert!(Citations::is_streamable());
    }

    #[test]
    fn its_description_is_the_prose_dspy_states() {
        let description = Citations::description().expect("a description");
        assert_eq!(description.name, "Citations");
        assert!(!description.replaces_schema);
        assert!(
            description
                .text
                .starts_with("Citations with quoted text and source references.")
        );
    }
}
