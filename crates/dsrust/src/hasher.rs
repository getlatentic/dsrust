//! dspy `Hasher` (`utils/hasher.py`): the digest `BootstrapFewShot` seeds a generator with.
//!
//! Vendored by dspy from HuggingFace's `datasets`, and used for one thing this crate reaches: a
//! predictor that answered more than once inside a single example produces more than one demo,
//! and only one is kept —
//!
//! ```text
//! rng = random.Random(Hasher.hash(tuple(demos)))
//! demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
//! ```
//!
//! so which demo a multi-hop program is taught by is decided by the sha256 of a *pickle*. Nothing
//! about that is incidental: change a byte of the pickle and a different demo survives, in about
//! half of cases. The writer that reproduces those bytes lives in this module’s `pickle`, which
//! documents what the format makes load-bearing: object identity, framing, and the opcode a size
//! selects.

use sha2::{Digest, Sha256};

use crate::Example;

mod pickle;

/// dspy `Hasher`.
pub struct Hasher;

impl Hasher {
    /// dspy `Hasher.hash_bytes`: one sha256 over the chunks, in order.
    pub fn hash_bytes(chunks: &[&[u8]]) -> String {
        let mut digest = Sha256::new();
        for chunk in chunks {
            digest.update(chunk);
        }
        hex(&digest.finalize())
    }

    /// dspy `Hasher.hash(tuple(demos))`: the digest of the pickled tuple.
    ///
    /// Upstream's signature is `hash(value: Any)`, and pickling `Any` is not something Rust can
    /// offer. The narrowing costs nothing in scope: the pinned tree calls `Hasher.hash` twice, and
    /// the other call is `clients/utils_finetune.py`'s, on the finetuning path this crate does not
    /// port. What reaches here is always a tuple of demos.
    pub fn hash(demos: &[Example]) -> String {
        Self::hash_bytes(&[&pickle::demos(demos)])
    }
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
