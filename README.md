# DsRust

Every production LLM call comes with the same plumbing: a prompt template, a JSON parser with a
retry loop, a reasoning pattern built by hand, and "tests" that amount to reading the output and
hoping. Switch models and you re-tune all of it.

DsRust replaces that with a declaration. You say what a step takes in and returns —
`question -> answer`, or a struct with typed fields — and DsRust builds the prompt, calls the
model, and parses the reply back into your types. Compose steps into a program; when you want it
better, hand it a metric and a few examples and let an optimizer search for the prompt and few-shot
examples that score highest.

It's a faithful Rust port of [DSPy](https://github.com/stanfordnlp/dspy) — to LLM engineering what
Tailwind is to CSS: a higher-level interface to the same power.

```rust
use dsrust::{predict, call};

let qa = predict!("question -> answer");
let out = call!(qa, question = "capital of France?").await?;
println!("{}", out.get("answer").unwrap());
```

```toml
[dependencies]
dsrust = "0.1.0-alpha.1"
```

> **Status: alpha.** The core — byte-level rendering and parsing, the optimizers, the RNG — is
> solid and already tested against DSPy's own suite. The API is real and growing toward full
> parity ([Roadmap](#roadmap)); pin an exact version while it fills in.

---

## Why DsRust, and not the other Rust one

There's already a Rust DSPy, and it's a rewrite — reimagined in idiomatic Rust, with prompts of its
own. DsRust makes the opposite bet: stay faithful down to the byte. The prompt it sends is the
prompt DSPy sends, and that isn't a claim in a README — DSPy's own test suite runs against DsRust's
renderer on every build.

That fidelity is the reason to reach for it:

- **You run the real thing, not a retelling.** DSPy's prompts, adapters, and its optimizers —
  MIPROv2, GEPA — come across exactly. Years of research, ported rather than paraphrased.
- **Compiled programs cross the language line.** Optimize one in DsRust and `dspy.load` opens it in
  Python; save one in Python and DsRust runs it. Same on-disk format, both directions.
- **Underneath, it's Rust.** Real parallelism instead of the GIL, no interpreter, one binary to
  ship. The model call is where the latency lives — but the rendering, parsing, parallel
  evaluation, and the optimizer's own search are all native.

|                       | DSPy (Python) | dspy-rs (Rust) | **DsRust (Rust)**                          |
|-----------------------|---------------|----------------|--------------------------------------------|
| Relationship to DSPy  | the original  | a rewrite      | **a faithful port**                        |
| Prompt bytes          | —             | its own        | **identical to DSPy, tested against its suite** |
| Compiled artifacts    | DSPy format   | its own        | **DSPy format — load in either**           |
| Runtime               | Python        | Rust           | Rust                                       |

**Full guide, with every module and the DSPy-vs-DsRust mapping side by side:
[`docs/usage.md`](docs/usage.md).**

---

## A tour

A task can be a typed struct, which gives you a reply checked at compile time:

```rust
use dsrust::{predict, call, Signature};

#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]  question: String,
    #[output] answer: String,
}

let out = call!(predict!(QA), question = "capital of France?").await?;
println!("{}", out.answer);   // typed, checked when this compiles
```

Reach a provider by naming it `provider/model-id` — the prefix is a **wire format**, not a brand,
so any OpenAI-compatible host is a base-URL away:

```rust
use dsrust::lm::{LM, configure};

configure(LM::new("openai/gpt-4o-mini")?);                 // from OPENAI_API_KEY

configure(                                                  // Groq, same wire
    LM::new("openai/llama-3.3-70b")?
        .with_openai_base_url("https://api.groq.com/openai/v1")
        .with_openai_key(std::env::var("GROQ_API_KEY")?),
);
```

`anthropic/…`, `ollama/…` and `ollama_chat/…` are the other built-in wires; your own provider is
the `ChatModel` trait. Compose and optimize:

```rust
let agent = ReAct::new(signature!("question -> answer"), tools);

let compiled = GEPA::new(metric, reflection_lm)   // evolves the prompt against your metric
    .with_max_metric_calls(200)
    .compile(program, trainset, valset)
    .await?;

compiled.save(std::path::Path::new("compiled.json"))?;   // DSPy's format — open it in Python
```

Run the whole loop — declare, ask, score, compile — with no provider and no API key:

```bash
cargo run --example quickstart
```

---

## How the fidelity claim is tested

Two layers, both against the pinned upstream (`dspy==3.3.0b1`, `gepa==0.1.1`, `optuna==4.9.0`),
never against a transcription of it:

1. **Committed goldens** (`tests/conformance/**`) — the exact bytes and decisions captured from
   *running* the pinned Python. `cargo test` checks against them with no Python needed.
2. **DSPy's own pytest suite, over DsRust** — a PyO3 bridge runs DSPy's actual tests (pinned in
   `third_party/dspy` as a submodule) with DsRust underneath. A crossing-counter fails any test
   that passes *without* touching the Rust crate.

Today: **~696 Rust tests**, **452 of DSPy's own tests passing through the crate**, plus
byte-verified reproductions of CPython's Mersenne Twister (checked against CPython's *own*
`test_guaranteed_stable` vector), numpy's RNG, optuna's TPE sampler, and the gepa engine.

```bash
git clone --recurse-submodules <repo>   # third_party/dspy at the pinned tag
cargo test --workspace                   # Rust suite + goldens (no Python)
uv sync && bash scripts/run_upstream_tests.sh   # DSPy's own tests, over DsRust
```

Details and the conformance ledger: [`docs/`](docs/) and `backlog.toml`.

---

## Roadmap

The byte and algorithm levels are strong — verified against DSPy's own tests. The API breadth is
filling in.

**Done**

- [x] **Adapters** — Chat, JSON, XML, BAML, TwoStep — byte-identical, incl. native function calling.
- [x] **Modules** — Predict, ChainOfThought, ReAct, MultiChainComparison, BestOfN, Refine, Parallel.
- [x] **Providers** — OpenAI-compatible, Anthropic, ollama; typed `ChatModel`/`LmRequest`/`LmResponse`,
  streaming, a response cache, litellm-grounded capability detection.
- [x] **Optimizers** — LabeledFewShot, BootstrapFewShot, COPRO, MIPROv2, GEPA (reflective mutation
  *and* merge, with CPython set-order and RNG reproduced).
- [x] **Save/load** in DSPy's on-disk format; `Reasoning`, `Tool`, `ToolCalls` + tool history.

**Next**

- [ ] **ReAct v2** — DSPy 3.3's native-tool-calling rewrite (current sprint).
- [ ] **Custom-type seam** — `dspy.Type` + `Image` / `Audio` / `File` / `Code`.

**Planned**

- [ ] `ProgramOfThought`, `CodeAct`, `RLM`; the remaining optimizers (SIMBA, KNNFewShot, …);
  retrieval; `History` as a type; a full error taxonomy.

---

## Related work & license

[krypticmouse/DSRs](https://github.com/krypticmouse/DSRs) (`dspy-rs`) is a DSPy *rewrite* for Rust
— an idiomatic reimagining. DsRust's distinct bet is fidelity: the DSPy you know, producing the
prompts DSPy produces, held to DSPy's own tests.

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

*DsRust is an independent port, not affiliated with or endorsed by the DSPy project or Stanford NLP.*
