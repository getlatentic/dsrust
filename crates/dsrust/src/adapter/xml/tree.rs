//! The XML a reply is read as, the way dspy 3.3.1 reads it: `xml.etree.ElementTree.fromstring`
//! over `<dspy_root>{completion}</dspy_root>`, which is expat underneath with namespace processing
//! on — so a prefixed name comes back as `{uri}local`, exactly as ElementTree spells it.
//!
//! A reply that is not well-formed is refused with expat's own words and position, because that
//! text reaches the caller (and the JSON fallback's log) as `Failed to parse XML: …`. Every position
//! rule was measured against expat rather than assumed; `tokenizer.rs` carries the cases.
use std::fmt;

/// One element: its name, its attributes in source order, and what it wraps, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) enum Node {
    Text(String),
    Element(Element),
}

impl Element {
    /// The element children, in order — `list(element)` upstream.
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|node| match node {
            Node::Element(element) => Some(element),
            Node::Text(_) => None,
        })
    }

    /// `element.text`: the text before the first child element, or nothing.
    pub fn text(&self) -> &str {
        match self.children.first() {
            Some(Node::Text(text)) => text,
            _ => "",
        }
    }

    /// `element.text + "".join(ET.tostring(child, encoding="unicode") for child in element)`: the
    /// inner XML as ElementTree re-serialises it — each child with its tail, so the whole body
    /// comes back.
    pub fn inner_xml(&self) -> String {
        let mut out = String::new();
        for node in &self.children {
            match node {
                Node::Text(text) => out.push_str(&escape_text(text)),
                Node::Element(element) => element.write(&mut out),
            }
        }
        out
    }

    /// `ET.tostring(element)` without the tail. Each call picks a prefix for every namespaced name
    /// in its subtree and declares them all on this element, sorted by prefix, ahead of the
    /// attributes; the numbering starts over with each call.
    fn write(&self, out: &mut String) {
        let names = QualifiedNames::over(self);
        self.write_with(out, &names, true);
    }

    fn write_with(&self, out: &mut String, names: &QualifiedNames, declares: bool) {
        let tag = names.of(&self.name);
        out.push('<');
        out.push_str(tag);
        if declares {
            for (uri, prefix) in names.declarations() {
                out.push_str(&format!(" xmlns:{prefix}=\"{}\"", escape_attribute(uri)));
            }
        }
        for (name, value) in &self.attributes {
            out.push_str(&format!(
                " {}=\"{}\"",
                names.of(name),
                escape_attribute(value)
            ));
        }
        if self.children.is_empty() {
            out.push_str(" />");
            return;
        }
        out.push('>');
        for node in &self.children {
            match node {
                Node::Text(text) => out.push_str(&escape_text(text)),
                Node::Element(element) => element.write_with(out, names, false),
            }
        }
        out.push_str("</");
        out.push_str(tag);
        out.push('>');
    }
}

/// ElementTree's `_namespaces`: a prefix for each `{uri}` met walking the subtree in document
/// order — tag first, then attribute names — a well-known one where the URI has it, else `nsN`
/// with N counting the URIs seen so far. `xml` is never declared.
struct QualifiedNames {
    qnames: Vec<(String, String)>,
    namespaces: Vec<(String, String)>,
}

impl QualifiedNames {
    fn over(element: &Element) -> Self {
        let mut names = Self {
            qnames: Vec::new(),
            namespaces: Vec::new(),
        };
        names.visit(element);
        names
    }

    fn visit(&mut self, element: &Element) {
        self.add(&element.name);
        for (name, _) in &element.attributes {
            self.add(name);
        }
        for child in element.elements() {
            self.visit(child);
        }
    }

    fn add(&mut self, qname: &str) {
        if self.qnames.iter().any(|(known, _)| known == qname) {
            return;
        }
        let Some((uri, local)) = qname
            .strip_prefix('{')
            .and_then(|rest| rest.rsplit_once('}'))
        else {
            self.qnames.push((qname.to_owned(), qname.to_owned()));
            return;
        };
        let prefix = match self.namespaces.iter().find(|(known, _)| known == uri) {
            Some((_, prefix)) => prefix.clone(),
            None => {
                let prefix = well_known(uri)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("ns{}", self.namespaces.len()));
                if prefix != "xml" {
                    self.namespaces.push((uri.to_owned(), prefix.clone()));
                }
                prefix
            }
        };
        self.qnames
            .push((qname.to_owned(), format!("{prefix}:{local}")));
    }

    fn of<'a>(&'a self, qname: &'a str) -> &'a str {
        self.qnames
            .iter()
            .find(|(known, _)| known == qname)
            .map_or(qname, |(_, spelled)| spelled.as_str())
    }

    fn declarations(&self) -> Vec<(&str, &str)> {
        let mut declarations: Vec<(&str, &str)> = self
            .namespaces
            .iter()
            .map(|(uri, prefix)| (uri.as_str(), prefix.as_str()))
            .collect();
        declarations.sort_by(|a, b| a.1.cmp(b.1));
        declarations
    }
}

/// ElementTree's `_namespace_map`: the prefixes it uses unasked.
fn well_known(uri: &str) -> Option<&'static str> {
    Some(match uri {
        "http://www.w3.org/XML/1998/namespace" => "xml",
        "http://www.w3.org/1999/xhtml" => "html",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#" => "rdf",
        "http://schemas.xmlsoap.org/wsdl/" => "wsdl",
        "http://www.w3.org/2001/XMLSchema" => "xs",
        "http://www.w3.org/2001/XMLSchema-instance" => "xsi",
        "http://purl.org/dc/elements/1.1/" => "dc",
        _ => return None,
    })
}

/// ElementTree's `_escape_cdata`.
fn escape_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// ElementTree's `_escape_attrib`: the text escapes, the quote, and each of the three whitespace
/// characters as a character reference.
fn escape_attribute(text: &str) -> String {
    escape_text(text)
        .replace('"', "&quot;")
        .replace('\r', "&#13;")
        .replace('\n', "&#10;")
        .replace('\t', "&#09;")
}

/// Where expat is in the text: the line 1-based, the column 0-based and counted in characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Position {
    pub line: usize,
    pub column: usize,
}

/// expat's refusal: its words, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct ParseError {
    pub kind: &'static str,
    pub at: Position,
}

impl fmt::Display for ParseError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "{}: line {}, column {}",
            self.kind, self.at.line, self.at.column
        )
    }
}

pub(super) const MISMATCHED: &str = "mismatched tag";
pub(super) const INVALID: &str = "not well-formed (invalid token)";
pub(super) const UNDEFINED_ENTITY: &str = "undefined entity";
pub(super) const JUNK: &str = "junk after document element";
pub(super) const UNCLOSED: &str = "unclosed token";
pub(super) const UNCLOSED_CDATA: &str = "unclosed CDATA section";
pub(super) const NO_ELEMENT: &str = "no element found";
pub(super) const BAD_CHARACTER_REFERENCE: &str = "reference to invalid character number";
pub(super) const DUPLICATE_ATTRIBUTE: &str = "duplicate attribute";
pub(super) const UNBOUND_PREFIX: &str = "unbound prefix";
pub(super) const RESERVED_XMLNS: &str =
    "reserved prefix (xmlns) must not be declared or undeclared";
pub(super) const RESERVED_XML: &str =
    "reserved prefix (xml) must not be undeclared or bound to another namespace name";
pub(super) const UNDECLARE: &str = "must not undeclare prefix";
pub(super) const XML_DECLARATION: &str = "XML or text declaration not at start of entity";

#[cfg(test)]
mod tests {
    use super::super::tokenizer::parse_root;

    fn inner(completion: &str) -> String {
        parse_root(completion).expect("parses").inner_xml()
    }

    /// Each string is `(root.text or "") + "".join(ET.tostring(c, encoding="unicode") for c in root)`
    /// copied from Python 3.13 over the same reply.
    #[test]
    fn inner_xml_is_element_trees_serialisation() {
        assert_eq!(
            inner("<reasoning><reasoning>inner</reasoning></reasoning>"),
            "<reasoning><reasoning>inner</reasoning></reasoning>"
        );
        assert_eq!(
            inner("a<b x=\"1&amp;2\">t</b>tail<e/>"),
            "a<b x=\"1&amp;2\">t</b>tail<e />"
        );
        assert_eq!(
            inner("<q a='&#10;&#9;&#13;\"&lt;&gt;&amp;'>t&lt;&amp;&gt;&quot;</q>tail&lt;"),
            "<q a=\"&#10;&#09;&#13;&quot;&lt;&gt;&amp;\">t&lt;&amp;&gt;\"</q>tail&lt;"
        );
        assert_eq!(inner("<q><![CDATA[<x>]]></q>"), "<q>&lt;x&gt;</q>");
        assert_eq!(inner("<q/><q></q>"), "<q /><q />");
        assert_eq!(inner("<a>&#13;</a>"), "<a>\r</a>");
        assert_eq!(
            inner("<a b='&#13;&#10;&#9;'/>"),
            "<a b=\"&#13;&#10;&#09;\" />"
        );
    }

    /// Namespaces are re-declared per top-level child, numbered from `ns0` again each time, a
    /// well-known URI keeping its prefix and still counting.
    #[test]
    fn namespaced_names_come_back_with_element_trees_prefixes() {
        assert_eq!(
            inner("<x xmlns='u'><q a='1'/></x>"),
            "<ns0:x xmlns:ns0=\"u\"><ns0:q a=\"1\" /></ns0:x>"
        );
        assert_eq!(
            inner("<x xmlns:p='u' p:a='1'><p:q/><r xmlns:p='v'><p:s/></r></x>"),
            "<x xmlns:ns0=\"u\" xmlns:ns1=\"v\" ns0:a=\"1\"><ns0:q /><r><ns1:s /></r></x>"
        );
        assert_eq!(
            inner("<q xmlns='u'><s/></q><z xmlns='u'/>"),
            "<ns0:q xmlns:ns0=\"u\"><ns0:s /></ns0:q><ns0:z xmlns:ns0=\"u\" />"
        );
        assert_eq!(inner("<x xml:lang='en'/>"), "<x xml:lang=\"en\" />");
        assert_eq!(
            inner(
                "<r xmlns:h='http://www.w3.org/1999/xhtml' xmlns:o='other'><h:div><o:x/></h:div></r>"
            ),
            "<r xmlns:html=\"http://www.w3.org/1999/xhtml\" xmlns:ns1=\"other\"><html:div><ns1:x /></html:div></r>"
        );
    }
}
