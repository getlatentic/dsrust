//! XML namespace processing as ElementTree's parser has it on: an `xmlns` attribute binds, a
//! prefixed name resolves to `{uri}local`, and the five refusals expat makes here are all reported
//! at the tag's `<` — measured, like the rest.
use super::tree::{
    DUPLICATE_ATTRIBUTE, ParseError, Position, RESERVED_XML, RESERVED_XMLNS, UNBOUND_PREFIX,
    UNDECLARE,
};

pub(super) const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";

/// A name as written: `NCName` or `prefix:NCName`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QName {
    pub prefix: Option<String>,
    pub local: String,
}

impl QName {
    /// The spelling in the text, which is what an end tag has to match.
    pub fn raw(&self) -> String {
        match &self.prefix {
            Some(prefix) => format!("{prefix}:{}", self.local),
            None => self.local.clone(),
        }
    }
}

/// One frame per open element: the prefixes it declared, `""` for the default namespace, and the
/// URI each binds — `None` where `xmlns=""` undeclares the default.
#[derive(Default)]
pub(super) struct Scope {
    frames: Vec<Vec<(String, Option<String>)>>,
}

impl Scope {
    /// The element's declarations taken up, then its attributes and its name resolved, in expat's
    /// order: a duplicate left after resolution is reported before an unbound element prefix.
    pub fn enter(
        &mut self,
        at: Position,
        name: &QName,
        attributes: Vec<(QName, String)>,
    ) -> Result<(String, Vec<(String, String)>), ParseError> {
        let refuse = |kind| ParseError { kind, at };
        let mut frame = Vec::new();
        let mut plain = Vec::new();
        for (attribute, value) in attributes {
            match (attribute.prefix.as_deref(), attribute.local.as_str()) {
                (None, "xmlns") => {
                    frame.push((String::new(), (!value.is_empty()).then_some(value)))
                }
                (Some("xmlns"), "xmlns") => return Err(refuse(RESERVED_XMLNS)),
                (Some("xmlns"), "xml") if value != XML_NS => return Err(refuse(RESERVED_XML)),
                (Some("xmlns"), "xml") => {}
                (Some("xmlns"), _) if value.is_empty() => return Err(refuse(UNDECLARE)),
                (Some("xmlns"), prefix) => frame.push((prefix.to_owned(), Some(value))),
                _ => plain.push((attribute, value)),
            }
        }
        self.frames.push(frame);
        let mut resolved: Vec<(String, String)> = Vec::new();
        for (attribute, value) in plain {
            let key = match &attribute.prefix {
                None => attribute.local.clone(),
                Some(prefix) => self
                    .expand(prefix, &attribute.local)
                    .ok_or_else(|| refuse(UNBOUND_PREFIX))?,
            };
            if resolved.iter().any(|(known, _)| *known == key) {
                return Err(refuse(DUPLICATE_ATTRIBUTE));
            }
            resolved.push((key, value));
        }
        let name = match &name.prefix {
            Some(prefix) => self
                .expand(prefix, &name.local)
                .ok_or_else(|| refuse(UNBOUND_PREFIX))?,
            None => match self.lookup("").flatten() {
                Some(uri) => format!("{{{uri}}}{}", name.local),
                None => name.local.clone(),
            },
        };
        Ok((name, resolved))
    }

    pub fn leave(&mut self) {
        self.frames.pop();
    }

    /// `{uri}local` for a bound prefix; `xml` is bound without a declaration.
    fn expand(&self, prefix: &str, local: &str) -> Option<String> {
        let uri = match prefix {
            "xml" => XML_NS.to_owned(),
            _ => self.lookup(prefix)??,
        };
        Some(format!("{{{uri}}}{local}"))
    }

    /// The innermost declaration of `prefix`, or none.
    fn lookup(&self, prefix: &str) -> Option<Option<String>> {
        self.frames.iter().rev().find_map(|frame| {
            frame
                .iter()
                .rev()
                .find(|(declared, _)| declared == prefix)
                .map(|(_, uri)| uri.clone())
        })
    }
}
