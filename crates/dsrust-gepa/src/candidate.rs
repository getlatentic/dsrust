//! gepa's `Candidate`: a component name to the text that component currently carries.
//!
//! **Insertion-ordered, not sorted.** gepa's is a Python `dict` and dspy builds it as
//! `{name: pred.signature.instructions for name, pred in student.named_predictors()}` — so the
//! order is the program's *declaration* order, and three places read it:
//!
//!   - the round-robin walk, which is `list_of_named_predictors` and decides which component the
//!     very first reflection rewrites;
//!   - `ComponentSelection::All`, whose order reaches the proposer;
//!   - the merge predicate, which is gepa's `list(program_candidates[ancestor].keys())`.
//!
//! A sorted map agrees with all three only when a program's predictors happen to be declared in
//! alphabetical order. A program declared `write, plan` reflects on `plan` first under a `BTreeMap`
//! and on `write` first under gepa.
//!
//! Hand-rolled over a `Vec` rather than pulling in `indexmap`, for the reason [`crate::pyset`] is
//! hand-rolled: the ordering *is* the thing being reproduced, so it belongs somewhere it can be read
//! against the Python rather than delegated to a crate whose guarantees are its own.

/// A component name to its text — gepa's `dict[str, str]`, keeping insertion order.
#[derive(Clone, Debug, Default)]
pub struct Candidate {
    entries: Vec<(String, String)>,
}

impl Candidate {
    /// An empty candidate.
    pub fn new() -> Self {
        Self::default()
    }

    /// The text a component carries, if it has one.
    pub fn get(&self, component: &str) -> Option<&String> {
        self.entries
            .iter()
            .find(|(name, _)| name == component)
            .map(|(_, text)| text)
    }

    /// Set a component's text, returning what it held before.
    ///
    /// A component already present keeps its position, as assigning to an existing key in a Python
    /// dict does — otherwise a merge that rewrites one component would move it to the end and
    /// change every later round-robin pick.
    pub fn insert(
        &mut self,
        component: impl Into<String>,
        text: impl Into<String>,
    ) -> Option<String> {
        let component = component.into();
        let text = text.into();
        match self.entries.iter_mut().find(|(name, _)| *name == component) {
            Some((_, held)) => Some(std::mem::replace(held, text)),
            None => {
                self.entries.push((component, text));
                None
            }
        }
    }

    /// Whether a component is present.
    pub fn contains_key(&self, component: &str) -> bool {
        self.get(component).is_some()
    }

    /// The component names, in declaration order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &String> {
        self.entries.iter().map(|(name, _)| name)
    }

    /// The component texts, in declaration order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &String> {
        self.entries.iter().map(|(_, text)| text)
    }

    /// Each component and its text, in declaration order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&String, &String)> {
        self.entries.iter().map(|(name, text)| (name, text))
    }

    /// How many components this candidate carries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it carries none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Extend<(String, String)> for Candidate {
    fn extend<T: IntoIterator<Item = (String, String)>>(&mut self, entries: T) {
        for (component, text) in entries {
            self.insert(component, text);
        }
    }
}

impl FromIterator<(String, String)> for Candidate {
    fn from_iter<T: IntoIterator<Item = (String, String)>>(entries: T) -> Self {
        let mut candidate = Self::new();
        candidate.extend(entries);
        candidate
    }
}

impl<const N: usize> From<[(String, String); N]> for Candidate {
    fn from(entries: [(String, String); N]) -> Self {
        entries.into_iter().collect()
    }
}

impl<'a> IntoIterator for &'a Candidate {
    type Item = (&'a String, &'a String);
    type IntoIter = std::vec::IntoIter<(&'a String, &'a String)>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

impl IntoIterator for Candidate {
    type Item = (String, String);
    type IntoIter = std::vec::IntoIter<(String, String)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// `candidate["name"]`, as gepa writes it. Panics on a component that is not there, which is
/// upstream's `KeyError`: every candidate in a merge is asserted to carry the same components, so
/// a miss is a broken invariant rather than something to handle.
impl std::ops::Index<&str> for Candidate {
    type Output = String;

    fn index(&self, component: &str) -> &String {
        self.get(component)
            .unwrap_or_else(|| panic!("candidate has no component {component:?}"))
    }
}

/// Order-independent, as Python's `dict.__eq__` is: two candidates carrying the same components
/// with the same texts are equal however they were built. What order *does* decide is which
/// component is rewritten next, and that is read through [`Candidate::keys`], never through this.
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(name, text)| other.get(name).is_some_and(|held| held == text))
    }
}

impl Eq for Candidate {}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(pairs: &[(&str, &str)]) -> Candidate {
        pairs
            .iter()
            .map(|(name, text)| ((*name).to_owned(), (*text).to_owned()))
            .collect()
    }

    /// The point of the type: a program declared `write, plan` walks `write` first, where a sorted
    /// map would walk `plan` first and reflect on the wrong component on iteration one.
    #[test]
    fn the_walk_is_declaration_order_not_alphabetical() {
        let candidate = built(&[("write", "a"), ("plan", "b"), ("check", "c")]);
        let names: Vec<&str> = candidate.keys().map(String::as_str).collect();
        assert_eq!(names, ["write", "plan", "check"]);
    }

    /// Assigning to a component already present keeps its position, as a Python dict does. A merge
    /// rewrites components in place, and moving one to the end would shift every later pick.
    #[test]
    fn rewriting_a_component_leaves_it_where_it_was() {
        let mut candidate = built(&[("write", "a"), ("plan", "b")]);
        assert_eq!(candidate.insert("write", "rewritten"), Some("a".to_owned()));
        let names: Vec<&str> = candidate.keys().map(String::as_str).collect();
        assert_eq!(names, ["write", "plan"]);
        assert_eq!(
            candidate.get("write").map(String::as_str),
            Some("rewritten")
        );
    }

    /// Equality ignores order, as Python's does — the order is read for the walk, not for identity.
    #[test]
    fn two_candidates_with_the_same_components_are_equal_however_they_were_built() {
        assert_eq!(
            built(&[("write", "a"), ("plan", "b")]),
            built(&[("plan", "b"), ("write", "a")])
        );
    }
}
