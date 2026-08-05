//! Where the cursor is, which is what most of the repair heuristics branch on.
//!
//! `current` is the innermost context and `context` the whole stack, and the two are read
//! separately — a quote inside an array nested in an object value is judged by the stack, while
//! whether a comma ends a member is judged by the innermost frame alone.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContextValue {
    ObjectKey,
    ObjectValue,
    Array,
}

pub(crate) struct JsonContext {
    stack: Vec<ContextValue>,
    pub(crate) current: Option<ContextValue>,
    pub(crate) empty: bool,
}

impl JsonContext {
    pub(crate) fn new() -> Self {
        Self {
            stack: Vec::new(),
            current: None,
            empty: true,
        }
    }

    pub(crate) fn set(&mut self, value: ContextValue) {
        self.stack.push(value);
        self.current = Some(value);
        self.empty = false;
    }

    pub(crate) fn reset(&mut self) {
        self.stack.pop();
        self.current = self.stack.last().copied();
        self.empty = self.stack.is_empty();
    }

    pub(crate) fn clear(&mut self) {
        self.stack.clear();
        self.current = None;
        self.empty = true;
    }

    /// `value in self.context`: anywhere in the stack, not only innermost.
    pub(crate) fn contains(&self, value: ContextValue) -> bool {
        self.stack.contains(&value)
    }

    pub(crate) fn is(&self, value: ContextValue) -> bool {
        self.current == Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resetting_uncovers_the_frame_below_rather_than_emptying() {
        let mut context = JsonContext::new();
        context.set(ContextValue::Array);
        context.set(ContextValue::ObjectValue);
        assert!(context.is(ContextValue::ObjectValue));
        assert!(context.contains(ContextValue::Array));
        context.reset();
        assert!(context.is(ContextValue::Array));
        assert!(!context.empty);
        context.reset();
        assert!(context.empty);
        assert_eq!(context.current, None);
    }
}
