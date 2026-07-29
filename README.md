# DsRust

DsRust is a Rust framework for building LLM features without hand-writing prompts, and a faithful
port of [DSPy](https://github.com/stanfordnlp/dspy). It works in three steps:

1. **Declare** a task by its inputs and outputs, like `question -> answer`. DsRust builds the
   prompt, calls the model, and gives you back typed values. No template, no JSON parsing, no retry
   loop.
2. **Wrap** it in a module to use a proven prompting technique without building it yourself:
   `ChainOfThought`, `ReAct` (with tools), and more.
3. **Optimize** on your own data. You create a metric and provide a few examples, and DsRust tunes
   the prompt to score best on your metric.

## Install

```bash
cargo add dsrust@0.1.0-alpha.2
cargo add tokio --features macros,rt-multi-thread
```

Name the version: cargo does not pick a pre-release on its own. Asking a model is a network call,
so every call is `async` and your program needs a runtime. Tokio is the only other crate you add,
and the flavour is yours to choose — DsRust names no tokio type, so it does not pick one for you.

## A whole program

```rust
use dsrust::lm::{LM, configure};
use dsrust::{call, predict};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure(LM::new("openai/gpt-4o-mini")?);   // reads OPENAI_API_KEY

    let qa = predict!("question -> answer");
    let out = call!(qa, question = "What is the capital of France?").await?;
    println!("{}", out.get("answer").unwrap().as_str().unwrap());
    Ok(())
}
```

```console
$ cargo run
Paris
```

A field arrives as JSON, so `as_str` is what turns `"Paris"` into `Paris`. Declare the task as a
struct instead and the field is already a `String` — see [`docs/usage.md`](docs/usage.md).

The run above used a local `llama-server` rather than OpenAI. Point `LM::new` at any
OpenAI-compatible host to do the same:

```rust
configure(
    LM::new("openai/gemma-3-1b")?
        .with_openai_base_url("http://127.0.0.1:8080/v1")
        .with_openai_key("not-needed-locally"),
);
```

> **Status: alpha.** The prompt bytes, the parsing, the optimizers and the RNG are tested against
> DSPy's own suite. The API is smaller than DSPy's and grows toward it ([Roadmap](#roadmap)).

---

## Why DsRust

Other Rust takes on DSPy are rewrites, reimagined in idiomatic Rust, with prompts of their own.
DsRust makes a different bet: stay faithful down to the byte. The prompt it sends is the prompt
DSPy sends, and that isn't a claim in a README. DSPy's own test suite runs against DsRust's
renderer on every build.

That fidelity is the reason to reach for it:

- **You run the real thing, not a retelling.** DSPy's prompts, adapters, and the optimizers MIPROv2 and GEPA come across exactly. Years of research, ported rather than paraphrased.
- **Compiled programs cross the language line.** Optimize one in DsRust and `dspy.load` opens it in
  Python; save one in Python and DsRust runs it. Same on-disk format, both directions.
- **Underneath, it's Rust.** Real parallelism instead of the GIL, no interpreter, one binary to
  ship. The model call is where the latency lives, but the rendering, parsing, parallel
  evaluation, and the optimizer's own search are all native.

|                    | **DsRust**                                        | Other Rust ports          |
|--------------------|---------------------------------------------------|---------------------------|
| Approach           | **a faithful port of DSPy**                       | a rewrite (self-described) |
| Prompt bytes       | **identical to DSPy, checked against DSPy's own tests** | their own            |
| Compiled programs  | **load in Python via DSPy's on-disk format**      | their own                 |
| Runtime            | Rust                                              | Rust                      |

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

Reach a provider by naming it `provider/model-id`. The prefix is a **wire format**, not a brand,
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
let agent = react!("question -> answer", tools);              // ReAct: thought, tool, observation
let agent = react_v2!("question -> answer", tools);           // ReAct v2: the model calls tools natively

let compiled = GEPA::new(metric, reflection_lm)   // evolves the prompt against your metric
    .with_max_metric_calls(200)
    .compile(program, trainset, valset)
    .await?;

compiled.save(std::path::Path::new("compiled.json"))?;   // DSPy's format, open it in Python
```

Run the whole loop (declare, ask, score, compile) with no provider and no API key:

```bash
cargo run --example quickstart
```

---

## How the fidelity claim is tested

Two layers, both against the pinned upstream (`dspy==3.3.0b1`, `gepa==0.1.1`, `optuna==4.9.0`),
never against a transcription of it:

1. **Committed goldens** (`crates/dsrust/tests/conformance/**`), the exact bytes and decisions captured from
   *running* the pinned Python. `cargo test` checks against them with no Python needed.
2. **DSPy's own pytest suite, over DsRust**: a PyO3 bridge runs DSPy's actual tests (pinned in
   `third_party/dspy` as a submodule) with DsRust underneath. A crossing-counter fails any test
   that passes *without* touching the Rust crate.

<!-- status: generated by scripts/status.py from backlog.toml [status] -->

Today: **890 Rust tests pass**, and **858 of DSPy's own**, across 51 of its 86 test files. 441 of those cross into the Rust crate.

<!-- /status -->

Plus byte-verified reproductions of CPython's Mersenne Twister (checked against CPython's *own*
`test_guaranteed_stable` vector), numpy's RNG, optuna's TPE sampler, and the gepa engine.

```bash
git clone --recurse-submodules <repo>   # third_party/dspy at the pinned tag
cargo test --workspace                   # Rust suite + goldens (no Python)
uv sync && bash scripts/run_upstream_tests.sh   # DSPy's own tests, over DsRust
```

Details and the conformance ledger: [`docs/`](docs/) and `backlog.toml`.

---

## Roadmap

DSPy's own tests check the bytes and the algorithms. Every public symbol DSPy defines in a ported
module is either mapped to a Rust one or carries a written reason for the difference.

**Done**

- [x] **Adapters**: Chat, JSON, XML, BAML, TwoStep. Byte-identical, incl. native function calling.
- [x] **Modules**: Predict, ChainOfThought, ReAct, ReActV2, MultiChainComparison, BestOfN, Refine,
  Parallel, ProgramOfThought, CodeAct, RLM.
- [x] **Providers**: OpenAI-compatible (Chat *and* Responses), Anthropic, ollama; typed
  `ChatModel`/`LmRequest`/`LmResponse`, streaming, a response cache, litellm-grounded capability
  detection.
- [x] **Optimizers**: LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch, Ensemble,
  COPRO, MIPROv2, GEPA (reflective mutation *and* merge, with CPython set-order and RNG reproduced),
  BetterTogether.
- [x] **Save/load** in DSPy's on-disk format; `Reasoning`, `Tool`, `ToolCalls`, `History` + tool
  history; native function calling and native reasoning (`reasoning_effort`).
- [x] **ReAct v2**: DSPy 3.3's native-tool-calling agent, its per-turn signature byte-conformant.
- [x] **Custom-type seam**: `dspy.Type` + `Image` / `Audio` / `File` / `Code` / `Citations` /
  `Document` — and your own type on the same trait.

**Next**

- [ ] **The typed LM boundary, finished**: adapters answer with `LmMessage` rather than the older
  `ChatTurn`. The last slice of the DSPy 3.3 type migration, and the one breaking change left.

**Planned**

- [ ] **A sandbox in the box.** The protocol is ported: `CodeInterpreter` carries `execute(code,
  variables)`, `start` and `shutdown`, beside `SandboxSerializable` and the REPL types that reach a
  prompt. DSPy's own Deno-sandbox tests for `ProgramOfThought` and `CodeAct` run against DsRust's
  loop, with the real sandbox running the model's code. What is missing is an implementation behind
  the trait. Today a caller supplies one.
- [ ] Retrieval and the optimizers that need it (KNN, KNNFewShot, SIMBA); finetuning and RL.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

*DsRust is an independent port, not affiliated with or endorsed by the DSPy project or Stanford NLP.*
