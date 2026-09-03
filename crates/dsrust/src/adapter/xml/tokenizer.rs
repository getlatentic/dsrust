//! Reading `<dspy_root>{completion}</dspy_root>` the way expat does, refusal positions included.
//!
//! Every rule was measured on the pinned Python: the character classes are `names.rs`, and each
//! refusal carries the position expat reports — the character that broke a token, the `<` of a
//! tag that never closed or that could not bind a prefix, the `&` of a reference, the name of a
//! mismatched end tag, the end of the text for a CDATA section that never closed.
use super::cursor::is_whitespace;
use super::names::{CHAR, NAME_CHAR, NAME_START, contains};
use super::namespaces::{QName, Scope};
use super::tree::{
    BAD_CHARACTER_REFERENCE, DUPLICATE_ATTRIBUTE, Element, INVALID, JUNK, MISMATCHED, NO_ELEMENT,
    Node, ParseError, Position, UNCLOSED, UNCLOSED_CDATA, UNDEFINED_ENTITY, XML_DECLARATION,
};

/// `ET.fromstring(f"<dspy_root>{completion}</dspy_root>")`: the root element, or expat's refusal.
pub(in super::super) fn parse_root(completion: &str) -> Result<Element, ParseError> {
    Parser::new(&format!("<dspy_root>{completion}</dspy_root>")).document()
}

pub(super) struct Parser {
    pub(super) chars: Vec<char>,
    pub(super) at: usize,
    pub(super) position: Position,
    scope: Scope,
}

impl Parser {
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            at: 0,
            position: Position { line: 1, column: 0 },
            scope: Scope::default(),
        }
    }

    fn document(mut self) -> Result<Element, ParseError> {
        let root = self.element()?;
        self.skip_whitespace();
        match self.peek() {
            None => Ok(root),
            Some(_) => Err(self.here(JUNK)),
        }
    }

    /// An element from its `<` to its end tag.
    fn element(&mut self) -> Result<Element, ParseError> {
        let at = self.position;
        self.bump();
        let qname = self.qualified_name(at)?;
        let (attributes, closed) = self.attributes(at)?;
        let (name, attributes) = self.scope.enter(at, &qname, attributes)?;
        let mut element = Element {
            name,
            attributes,
            children: Vec::new(),
        };
        if !closed {
            self.body(&mut element, &qname.raw())?;
        }
        self.scope.leave();
        Ok(element)
    }

    /// The rest of a start tag after its name: `attr="…"` pairs, then `>` or `/>`.
    fn attributes(&mut self, at: Position) -> Result<(Vec<(QName, String)>, bool), ParseError> {
        let mut attributes: Vec<(QName, String)> = Vec::new();
        loop {
            let spaced = self.skip_whitespace();
            match self.peek() {
                None => return Err(ParseError { kind: UNCLOSED, at }),
                Some('>') => {
                    self.bump();
                    return Ok((attributes, false));
                }
                Some('/') => {
                    self.bump();
                    match self.peek() {
                        Some('>') => self.bump(),
                        None => return Err(ParseError { kind: UNCLOSED, at }),
                        Some(_) => return Err(self.here(INVALID)),
                    };
                    return Ok((attributes, true));
                }
                Some(c) if spaced && contains(NAME_START, c) => {
                    let name_at = self.position;
                    let name = self.qualified_name(at)?;
                    if attributes.iter().any(|(known, _)| *known == name) {
                        return Err(ParseError {
                            kind: DUPLICATE_ATTRIBUTE,
                            at: name_at,
                        });
                    }
                    self.skip_whitespace();
                    match self.peek() {
                        Some('=') => self.bump(),
                        None => return Err(ParseError { kind: UNCLOSED, at }),
                        Some(_) => return Err(self.here(INVALID)),
                    };
                    self.skip_whitespace();
                    let value = self.attribute_value(at)?;
                    attributes.push((name, value));
                }
                Some(_) => return Err(self.here(INVALID)),
            }
        }
    }

    /// A quoted value, normalised: each literal tab, line end or return becomes one space, and a
    /// character reference stays the character it names.
    fn attribute_value(&mut self, tag: Position) -> Result<String, ParseError> {
        let quote = match self.peek() {
            Some(quote @ ('"' | '\'')) => quote,
            Some(_) => return Err(self.here(INVALID)),
            None => {
                return Err(ParseError {
                    kind: UNCLOSED,
                    at: tag,
                });
            }
        };
        self.bump();
        let mut value = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(ParseError {
                        kind: UNCLOSED,
                        at: tag,
                    });
                }
                Some(c) if c == quote => {
                    self.bump();
                    return Ok(value);
                }
                Some('<') => return Err(self.here(INVALID)),
                Some('&') => value.push_str(&self.reference(Some(tag))?),
                Some('\t' | '\r' | '\n') => {
                    self.bump();
                    value.push(' ');
                }
                Some(c) if contains(CHAR, c) => {
                    self.bump();
                    value.push(c);
                }
                Some(_) => return Err(self.here(INVALID)),
            }
        }
    }

    /// Everything between a start tag and its end tag, `raw` being the start tag's spelling.
    fn body(&mut self, element: &mut Element, raw: &str) -> Result<(), ParseError> {
        let mut text = String::new();
        loop {
            match self.peek() {
                None => return Err(self.here(NO_ELEMENT)),
                Some('<') if self.starts_with("</") => {
                    self.end_tag(raw)?;
                    if !text.is_empty() {
                        element.children.push(Node::Text(text));
                    }
                    return Ok(());
                }
                Some('<') if self.starts_with("<!") => self.markup_declaration(&mut text)?,
                Some('<') if self.starts_with("<?") => self.processing_instruction()?,
                Some('<') => {
                    if !text.is_empty() {
                        element.children.push(Node::Text(std::mem::take(&mut text)));
                    }
                    let child = self.element()?;
                    element.children.push(Node::Element(child));
                }
                Some('&') => text.push_str(&self.reference(None)?),
                Some(']') if self.starts_with("]]>") => {
                    self.bump();
                    self.bump();
                    return Err(self.here(INVALID));
                }
                Some(c) if contains(CHAR, c) => text.push(self.bump().unwrap_or(c)),
                Some(_) => return Err(self.here(INVALID)),
            }
        }
    }

    /// `</name>`, which has to spell the start tag's name; a mismatch is reported at the name.
    fn end_tag(&mut self, raw: &str) -> Result<(), ParseError> {
        let at = self.position;
        self.bump();
        self.bump();
        let name_at = self.position;
        let name = self.qualified_name(at)?;
        self.skip_whitespace();
        match self.peek() {
            Some('>') => self.bump(),
            None => return Err(ParseError { kind: UNCLOSED, at }),
            Some(_) => return Err(self.here(INVALID)),
        };
        match name.raw() == raw {
            true => Ok(()),
            false => Err(ParseError {
                kind: MISMATCHED,
                at: name_at,
            }),
        }
    }

    /// `<!--…-->` or `<![CDATA[…]]>`; anything else after `<!` is refused at the first character
    /// that does not spell one of them.
    fn markup_declaration(&mut self, text: &mut String) -> Result<(), ParseError> {
        let at = self.position;
        if self.chars.get(self.at + 2) == Some(&'[') {
            self.literal("<![CDATA[", at)?;
            return self.cdata(text);
        }
        self.literal("<!--", at)?;
        loop {
            match self.peek() {
                None => return Err(ParseError { kind: UNCLOSED, at }),
                Some('-') if self.starts_with("--") => {
                    self.bump();
                    self.bump();
                    return match self.peek() {
                        Some('>') => {
                            self.bump();
                            Ok(())
                        }
                        Some(_) => Err(self.here(INVALID)),
                        None => Err(ParseError { kind: UNCLOSED, at }),
                    };
                }
                Some(c) if contains(CHAR, c) => {
                    self.bump();
                }
                Some(_) => return Err(self.here(INVALID)),
            }
        }
    }

    /// The section's text, taken as it is up to `]]>`.
    fn cdata(&mut self, text: &mut String) -> Result<(), ParseError> {
        loop {
            if self.starts_with("]]>") {
                for _ in 0..3 {
                    self.bump();
                }
                return Ok(());
            }
            match self.peek() {
                None => return Err(self.here(UNCLOSED_CDATA)),
                Some(c) if contains(CHAR, c) => text.push(self.bump().unwrap_or(c)),
                Some(_) => return Err(self.here(INVALID)),
            }
        }
    }

    /// `<?target …?>`, skipped. A target spelling `xml` in any case is refused: the declaration
    /// where it is not at the start, an invalid token for the other spellings.
    fn processing_instruction(&mut self) -> Result<(), ParseError> {
        let at = self.position;
        self.bump();
        self.bump();
        let target = self.ncname(at)?;
        if target.eq_ignore_ascii_case("xml") {
            return Err(match target.as_str() {
                "xml" => ParseError {
                    kind: XML_DECLARATION,
                    at,
                },
                _ => self.here(INVALID),
            });
        }
        match self.peek() {
            Some('?') => {}
            Some(c) if is_whitespace(c) => {}
            Some(_) => return Err(self.here(INVALID)),
            None => return Err(ParseError { kind: UNCLOSED, at }),
        }
        loop {
            if self.starts_with("?>") {
                self.bump();
                self.bump();
                return Ok(());
            }
            match self.peek() {
                None => return Err(ParseError { kind: UNCLOSED, at }),
                Some(c) if contains(CHAR, c) => {
                    self.bump();
                }
                Some(_) => return Err(self.here(INVALID)),
            }
        }
    }

    /// `&name;`, `&#N;` or `&#xH;`, the `&` still ahead: the text it stands for. An entity nobody
    /// declared is reported at the `&` in content and at the tag in an attribute.
    fn reference(&mut self, tag: Option<Position>) -> Result<String, ParseError> {
        let amp = self.position;
        let token = tag.unwrap_or(amp);
        self.bump();
        if self.peek() == Some('#') {
            return self.character_reference(amp, token);
        }
        let mut name = String::new();
        while let Some(c) = self.peek().filter(|c| contains(NAME_CHAR, *c)) {
            name.push(c);
            self.bump();
        }
        match self.peek() {
            None => {
                return Err(ParseError {
                    kind: UNCLOSED,
                    at: token,
                });
            }
            Some(';') if !name.is_empty() => {}
            Some(_) => return Err(self.here(INVALID)),
        }
        self.bump();
        Ok(match name.as_str() {
            "amp" => "&",
            "lt" => "<",
            "gt" => ">",
            "quot" => "\"",
            "apos" => "'",
            _ => {
                return Err(ParseError {
                    kind: UNDEFINED_ENTITY,
                    at: tag.unwrap_or(amp),
                });
            }
        }
        .to_owned())
    }

    /// After the `&`: `#` then decimal digits, or `#x` then hex digits, then `;`. A number that is
    /// not an XML character is refused at the `&`.
    fn character_reference(
        &mut self,
        amp: Position,
        token: Position,
    ) -> Result<String, ParseError> {
        self.bump();
        let hex = self.peek() == Some('x');
        if hex {
            self.bump();
        }
        let mut digits = String::new();
        while let Some(c) = self.peek().filter(|c| {
            if hex {
                c.is_ascii_hexdigit()
            } else {
                c.is_ascii_digit()
            }
        }) {
            digits.push(c);
            self.bump();
        }
        match self.peek() {
            None => {
                return Err(ParseError {
                    kind: UNCLOSED,
                    at: token,
                });
            }
            Some(';') if !digits.is_empty() => {}
            Some(_) => return Err(self.here(INVALID)),
        }
        self.bump();
        let radix = if hex { 16 } else { 10 };
        match u32::from_str_radix(&digits, radix)
            .ok()
            .and_then(char::from_u32)
        {
            Some(c) if contains(CHAR, c) => Ok(c.to_string()),
            _ => Err(ParseError {
                kind: BAD_CHARACTER_REFERENCE,
                at: amp,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(completion: &str) -> String {
        parse_root(completion).expect_err("refused").to_string()
    }

    /// Each expectation is `ET.fromstring(f"<dspy_root>{completion}</dspy_root>")` on Python 3.13,
    /// copied from its `ParseError`; the wrapper puts every column 11 past the completion's own.
    #[test]
    fn expat_positions_for_tags() {
        assert_eq!(refusal("<a>x</b>"), "mismatched tag: line 1, column 17");
        assert_eq!(refusal("<a>éé</b>"), "mismatched tag: line 1, column 18");
        assert_eq!(refusal("<a>x"), "mismatched tag: line 1, column 17");
        assert_eq!(
            refusal("<a>x</a>\n<b>y</c>"),
            "mismatched tag: line 2, column 6"
        );
        assert_eq!(refusal("\t<a>x</b>"), "mismatched tag: line 1, column 18");
        assert_eq!(
            refusal("a < b <a>x</a>"),
            "not well-formed (invalid token): line 1, column 14"
        );
        assert_eq!(
            refusal("<3 <a>x</a>"),
            "not well-formed (invalid token): line 1, column 12"
        );
        assert_eq!(
            refusal("<€>x</€>"),
            "not well-formed (invalid token): line 1, column 12"
        );
        assert_eq!(
            refusal("<b€>x</b€>"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(
            refusal("< a>"),
            "not well-formed (invalid token): line 1, column 12"
        );
        assert_eq!(
            refusal("<a/ >"),
            "not well-formed (invalid token): line 1, column 14"
        );
        assert_eq!(
            refusal("<a"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(
            refusal("</"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(
            refusal("<a></></a>"),
            "not well-formed (invalid token): line 1, column 16"
        );
        assert_eq!(
            refusal("<:c/>"),
            "not well-formed (invalid token): line 1, column 12"
        );
        assert_eq!(
            refusal("<c:/>"),
            "not well-formed (invalid token): line 1, column 14"
        );
        assert_eq!(
            refusal("<x xmlns:a='u'><a:b:c/></x>"),
            "not well-formed (invalid token): line 1, column 30"
        );
        assert_eq!(
            refusal("</dspy_root><a>"),
            "junk after document element: line 1, column 23"
        );
    }

    #[test]
    fn expat_positions_for_attributes() {
        assert_eq!(
            refusal("<a b>x</a>"),
            "not well-formed (invalid token): line 1, column 15"
        );
        assert_eq!(
            refusal("<a b=1>x</a>"),
            "not well-formed (invalid token): line 1, column 16"
        );
        assert_eq!(
            refusal("<a b=1/>"),
            "not well-formed (invalid token): line 1, column 16"
        );
        assert_eq!(
            refusal("<a b='<'/>"),
            "not well-formed (invalid token): line 1, column 17"
        );
        assert_eq!(
            refusal("<a b='1'c='2'/>"),
            "not well-formed (invalid token): line 1, column 19"
        );
        assert_eq!(
            refusal("<a\u{a0}b='1'/>"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(
            refusal("<a b='"),
            "not well-formed (invalid token): line 1, column 17"
        );
        assert_eq!(
            refusal("<a b='1' b='2'/>"),
            "duplicate attribute: line 1, column 20"
        );
        assert_eq!(
            refusal("<a b='&nbsp;'/>"),
            "undefined entity: line 1, column 11"
        );
        assert_eq!(
            refusal("<a b='&#0;'/>"),
            "reference to invalid character number: line 1, column 17"
        );
    }

    #[test]
    fn expat_positions_for_content() {
        assert_eq!(
            refusal("a & b"),
            "not well-formed (invalid token): line 1, column 14"
        );
        assert_eq!(
            refusal("&amp y"),
            "not well-formed (invalid token): line 1, column 15"
        );
        assert_eq!(
            refusal("&"),
            "not well-formed (invalid token): line 1, column 12"
        );
        assert_eq!(refusal("a &nbsp; b"), "undefined entity: line 1, column 13");
        assert_eq!(
            refusal("<a>&#0;</a>"),
            "reference to invalid character number: line 1, column 14"
        );
        assert_eq!(
            refusal("<a>&#xD800;</a>"),
            "reference to invalid character number: line 1, column 14"
        );
        assert_eq!(
            refusal("<a>&#x110000;</a>"),
            "reference to invalid character number: line 1, column 14"
        );
        assert_eq!(
            refusal("<a>&#X41;</a>"),
            "not well-formed (invalid token): line 1, column 16"
        );
        assert_eq!(
            refusal("<a>&#;</a>"),
            "not well-formed (invalid token): line 1, column 16"
        );
        assert_eq!(
            refusal("<a>&#x;</a>"),
            "not well-formed (invalid token): line 1, column 17"
        );
        assert_eq!(
            refusal("<a>&#4x;</a>"),
            "not well-formed (invalid token): line 1, column 17"
        );
        assert_eq!(
            refusal("<a>&#41</a>"),
            "not well-formed (invalid token): line 1, column 18"
        );
        assert_eq!(
            refusal("<a>x\u{1}y</a>"),
            "not well-formed (invalid token): line 1, column 15"
        );
        assert_eq!(
            refusal("<a>\u{ffff}</a>"),
            "not well-formed (invalid token): line 1, column 14"
        );
        assert_eq!(
            refusal("<a>x]]>y</a>"),
            "not well-formed (invalid token): line 1, column 17"
        );
        assert_eq!(
            refusal("<a>x\r<b</a>"),
            "not well-formed (invalid token): line 2, column 2"
        );
        assert_eq!(
            refusal("<a>x\r\n<b</a>"),
            "not well-formed (invalid token): line 2, column 2"
        );
    }

    #[test]
    fn expat_positions_for_comments_sections_and_instructions() {
        assert_eq!(
            refusal("<!DOCTYPE x>"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(
            refusal("<!x>"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(
            refusal("<!-x"),
            "not well-formed (invalid token): line 1, column 14"
        );
        assert_eq!(refusal("<!-- x"), "unclosed token: line 1, column 11");
        assert_eq!(
            refusal("<!-- x -- y -->"),
            "not well-formed (invalid token): line 1, column 20"
        );
        assert_eq!(
            refusal("<!-- x --->"),
            "not well-formed (invalid token): line 1, column 20"
        );
        assert_eq!(
            refusal("<a><!--\r\n-- x-->"),
            "not well-formed (invalid token): line 2, column 2"
        );
        assert_eq!(
            refusal("<![CDAT x"),
            "not well-formed (invalid token): line 1, column 18"
        );
        assert_eq!(
            refusal("<![CDATA[ x"),
            "unclosed CDATA section: line 1, column 34"
        );
        assert_eq!(
            refusal("<? x"),
            "not well-formed (invalid token): line 1, column 13"
        );
        assert_eq!(refusal("<?x y"), "unclosed token: line 1, column 11");
        assert_eq!(
            refusal("<?xml version='1.0'?>"),
            "XML or text declaration not at start of entity: line 1, column 11"
        );
        assert_eq!(
            refusal("<?XML x?>"),
            "not well-formed (invalid token): line 1, column 16"
        );
    }

    #[test]
    fn expat_positions_for_namespaces() {
        assert_eq!(refusal("<x p:q='1'/>"), "unbound prefix: line 1, column 11");
        assert_eq!(refusal("<p:q/>"), "unbound prefix: line 1, column 11");
        assert_eq!(
            refusal("<a>\n <p:y/></a>"),
            "unbound prefix: line 2, column 1"
        );
        assert_eq!(
            refusal("<x xmlns:xmlns='u'/>"),
            "reserved prefix (xmlns) must not be declared or undeclared: line 1, column 11"
        );
        assert_eq!(
            refusal("<x xmlns:xml='other'/>"),
            "reserved prefix (xml) must not be undeclared or bound to another namespace name: line 1, column 11"
        );
        assert_eq!(
            refusal("<x xmlns:p=''/>"),
            "must not undeclare prefix: line 1, column 11"
        );
        assert_eq!(
            refusal("<x xmlns:p='u' xmlns:q='u' p:a='1' q:a='2'/>"),
            "duplicate attribute: line 1, column 11"
        );
    }

    /// Refusals the wrapper makes unreachable, measured on the bare text: the text running out
    /// inside any token is that token left unclosed, reported where it began.
    #[test]
    fn expat_positions_at_the_end_of_the_text() {
        let bare = |text: &str| {
            Parser::new(text)
                .document()
                .expect_err("refused")
                .to_string()
        };
        assert_eq!(bare("<a>xx"), "no element found: line 1, column 5");
        for text in [
            "<a><b",
            "<a><",
            "<a></",
            "<a></b",
            "<a></b ",
            "<a><?",
            "<a><?p",
            "<a>&",
            "<a>&#",
            "<a><b:",
            "<a><b c",
            "<a><b c=",
            "<a><b c='x",
            "<a><!-",
            "<a><![CDATA",
            "<a><b/",
            "<a><b ",
        ] {
            assert_eq!(bare(text), "unclosed token: line 1, column 3", "{text:?}");
        }
    }

    fn shape(element: &Element) -> String {
        let attributes: Vec<String> = element
            .attributes
            .iter()
            .map(|(k, v)| format!("{k}={v:?}"))
            .collect();
        let children: Vec<String> = element
            .children
            .iter()
            .map(|node| match node {
                Node::Text(text) => format!("{text:?}"),
                Node::Element(child) => shape(child),
            })
            .collect();
        format!(
            "{}[{}]({})",
            element.name,
            attributes.join(","),
            children.join(",")
        )
    }

    fn read(completion: &str) -> String {
        let root = parse_root(completion).expect("parses");
        root.children
            .iter()
            .map(|node| match node {
                Node::Text(text) => format!("{text:?}"),
                Node::Element(child) => shape(child),
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Each expectation is the tree Python 3.13 builds, spelled tag, attributes, then children.
    #[test]
    fn well_formed_replies_read_as_element_tree_reads_them() {
        assert_eq!(
            read("<!-- c --><a><![CDATA[<x>]]></a><b/><c />&#65;&#x42;&amp;"),
            "a[](\"<x>\"),b[](),c[](),\"AB&\""
        );
        assert_eq!(
            read("<a>&lt;&gt;&amp;&quot;&apos;&#65;&#x42;</a>"),
            "a[](\"<>&\\\"'AB\")"
        );
        assert_eq!(read("<a>x</a >"), "a[](\"x\")");
        assert_eq!(read("<a>t<!--c-->u<?p q?>v</a>"), "a[](\"tuv\")");
        assert_eq!(read("<a>1\r\n2\r3</a>"), "a[](\"1\\n2\\n3\")");
        assert_eq!(read("<a><![CDATA[1\r\n2]]></a>"), "a[](\"1\\n2\")");
        assert_eq!(read("<a b='x\r\ny\tz'/>"), "a[b=\"x y z\"]()");
        assert_eq!(read("<a b='&#13;&#10;&#9;'/>"), "a[b=\"\\r\\n\\t\"]()");
        assert_eq!(
            read("<x xmlns='u'><q a='1'/></x>"),
            "{u}x[]({u}q[a=\"1\"]())"
        );
        assert_eq!(
            read("<x xmlns:p='u' p:a='1'><p:q/><r xmlns:p='v'><p:s/></r></x>"),
            "x[{u}a=\"1\"]({u}q[](),r[]({v}s[]()))"
        );
        assert_eq!(
            read("<x xml:lang='en'/>"),
            "x[{http://www.w3.org/XML/1998/namespace}lang=\"en\"]()"
        );
        assert_eq!(
            read("<x xmlns='u'><y xmlns=''><q/></y></x>"),
            "{u}x[](y[](q[]()))"
        );
        assert_eq!(
            read("<x><entry key='k'>v</entry></x>"),
            "x[](entry[key=\"k\"](\"v\"))"
        );
        assert_eq!(
            read("<xְ><answer>Paris</answer></xְ>"),
            "xְ[](answer[](\"Paris\"))"
        );
    }
}
