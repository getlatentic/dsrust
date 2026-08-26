//! Whether a saved program came from somewhere the caller vouches for.
//!
//! dspy's `allow_unsafe_lm_state`, which is one keyword on `load` and `load_state`. It is a type
//! here rather than a `bool` because what it widens is not obvious from a `true`, and because the
//! two sides of it have different consequences: one is about *where a call goes*, the other about
//! *what code runs*. Upstream's single flag controls both.

/// How much of a saved `lm` block to believe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    /// What every ordinary load gets, and dspy's default.
    ///
    /// `api_base`, `base_url` and `model_list` are dropped: they decide which host answers, and a
    /// compiled program is something people pass around. Everything else in the block — the model
    /// name, the sampling settings — is kept and honoured.
    Default,
    /// The caller states this file came from somewhere they trust.
    ///
    /// The three redirect keys survive, so a program compiled against a private endpoint reaches
    /// it again. This does **not** make a custom LM class loadable: dspy's flag also permits
    /// importing the class the block names, and a Rust binary has no importer at all. A block
    /// naming one fails either way, with a message saying so.
    File,
}

impl Trust {
    /// Whether a block's redirect keys survive the load.
    pub fn allows_redirect(self) -> bool {
        matches!(self, Trust::File)
    }
}
