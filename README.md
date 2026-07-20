# dsrs — DSPy in Rust

A Rust implementation of [DSPy](https://github.com/stanfordnlp/dspy): signatures, adapters,
modules, and — as they land — evaluation and optimizers.

The goal is **fidelity, not inspiration**. Where DSPy renders a prompt, this crate renders the
same bytes. That is a claim worth testing rather than asserting, so upstream's own adapter
tests are the acceptance criteria: `tests/conformance/` holds cases lifted from DSPy's
`format_exact_messages_*` suite with the expected messages copied verbatim, and
`tests/conformance.rs` runs this crate's renderer against them.

```bash
cargo test --test conformance   # are we still Python DSPy?
cargo test                      # everything
```

## Status

| Layer | State |
|---|---|
| Signatures (`#[derive(Signature)]`, typed + complex fields) | built |
| `Adapter` trait: `ChatAdapter`, `JsonAdapter`, or your own | built, conformance-checked |
| `Predict`, `ChainOfThought` (+ typed forms) | built |
| LM layer, global config with per-call override | built |
| Two-tier feedback retry | built |
| `Example` / `Prediction` with provenance | next |
| `Evaluate` (dataset + metric + parallel) | planned |
| Optimizers (labeled few-shot → bootstrap → search) | planned |
| ReAct | planned |

## Why another one

[`dspy-rs`](https://github.com/krypticmouse/DSRs) covers similar ground with a larger
dependency surface (arrow, parquet, hf-hub, rig-core). This crate keeps the dependency list
short enough to embed in a service, and treats byte-level agreement with Python DSPy as the
thing being tested.
