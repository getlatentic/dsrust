# DsRust

A faithful, byte-for-byte Rust port of [DSPy](https://github.com/stanfordnlp/dspy): declare a task,
and the library writes the prompt, calls the model, and hands back typed values.

## Highlights

- **DSPy's prompts, not our own.** The bytes are identical, and DSPy's own pytest suite is what
  says so — it runs against DsRust's renderer. A
  [crossing counter](#how-the-fidelity-claim-is-tested) fails any test that passes without touching
  the Rust crate.
- **You declare a task, not a prompt.** `question -> answer` gives you the prompt, the call, the
  parsing and the retry. No template, no JSON handling.
- **Reuse a proven prompting technique** rather than reimplementing it: `ChainOfThought`, `ReAct`,
  `ReActV2`, `ProgramOfThought`, `CodeAct`, `RLM`, `BestOfN`, `Refine`.
- **Code-writing modules get a sandbox that ships.** `DenoInterpreter` runs DSPy's own `runner.js`
  under Deno and Pyodide, so generated code lands where DSPy's does. `deno` is the one prerequisite,
  as it is for DSPy.
- **Optimize against your own metric.** MIPROv2 and GEPA, each reproduced down to its RNG and search
  order, so a compile here makes the choices a compile there makes.
- **Compiled programs cross the language line.** `dspy.load` opens what DsRust saves, and DsRust
  runs what Python saved. Same on-disk format, both directions.
- **Any OpenAI-compatible host is a base URL away**: OpenAI, Groq, Together, vLLM, LM Studio,
  `llama-server`. Anthropic and ollama have their own wires, and `ChatModel` is yours.
- **Rust underneath.** Real parallelism, no interpreter, one binary. Rendering, parsing, parallel
  evaluation and the optimizer's search are all native.

## Installation

```bash
cargo add dsrust@0.1.0-alpha.2
cargo add tokio --features macros,rt-multi-thread
```

Name the version: cargo does not pick a pre-release on its own. Every call is a network call, so
every call is `async` and your program needs a runtime. The flavour is yours to choose — DsRust
names no tokio type, so it does not pick a scheduler for you.

> **Status: alpha.** DSPy's own suite tests the prompt bytes, the parsing, the optimizers and the
> RNG. The API is smaller than DSPy's and grows toward it ([Roadmap](#roadmap)).

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

That run used a local `llama-server`; point `LM::new` at any OpenAI-compatible host to do the same.

A field arrives as JSON, so `as_str` is what turns `"Paris"` into `Paris`. Declare the task as a
struct and the field is a `String` already.

## Documentation

[`docs/usage.md`](docs/usage.md) is the full guide: every module, every declaration spelling, and
the DSPy-vs-DsRust mapping side by side.

## Features

### A task is a string or a struct

```rust
let qa = predict!("question -> answer");                       // names only
let m  = predict!("question: str, context: list[str] -> answer: int");   // types too
```

```rust
#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]  question: String,
    #[output] answer: String,
}

let out = call!(predict!(QA), question = "capital of France?").await?;
println!("{}", out.answer);   // a String, checked when this compiles
```

A bad signature fails the build. `predict!("a -> b -> c")` is a compile error carrying DSPy's own
message, pointed at the literal; DSPy raises the equivalent when the program runs.

### Reaching a provider

The prefix is a **wire format**, not a brand, so a host is a base URL rather than a new provider.

```rust
configure(LM::new("openai/gpt-4o-mini")?);                  // OPENAI_API_KEY

configure(                                                   // Groq, same wire
    LM::new("openai/llama-3.3-70b")?
        .with_openai_base_url("https://api.groq.com/openai/v1")
        .with_openai_key(std::env::var("GROQ_API_KEY")?),
);

configure(                                                   // a local llama-server
    LM::new("openai/gemma-3-1b")?
        .with_openai_base_url("http://127.0.0.1:8080/v1")
        .with_openai_key("not-needed-locally"),
);
```

`anthropic/…`, `ollama/…` and `ollama_chat/…` are the other built-in wires. Your own provider is the
`ChatModel` trait.

### Optimizing, and saving what it found

```rust
let compiled = GEPA::new(metric, reflection_lm)   // evolves the prompt against your metric
    .with_max_metric_calls(200)
    .compile(program, trainset, valset)
    .await?;

compiled.save(std::path::Path::new("compiled.json"))?;   // DSPy's format; open it in Python
```

Run the whole loop — declare, ask, score, compile — with no provider and no API key:

```bash
cargo run --example quickstart
```

## How the fidelity claim is tested

Two layers, both against the pinned upstream (`dspy==3.3.0b1`, `gepa==0.1.1`, `optuna==4.9.0`),
never against a transcription of it:

1. **Committed goldens** (`crates/dsrust/tests/conformance/**`), the exact bytes and decisions
   captured from *running* the pinned Python. `cargo test` checks against them, no Python needed.
2. **DSPy's own pytest suite, over DsRust.** A PyO3 bridge runs DSPy's actual tests with DsRust
   underneath. A crossing counter fails any test that passes *without* touching the Rust crate.

<!-- status: generated by scripts/status.py from backlog.toml [status] -->

Today: **912 Rust tests pass**, and **891 of DSPy's own**, across 52 of its 86 test files. 479 of those cross into the Rust crate.

<!-- /status -->

Plus byte-verified reproductions of CPython's Mersenne Twister (against CPython's *own*
`test_guaranteed_stable` vector), numpy's RNG, optuna's TPE sampler, and the gepa engine.

A third check puts a recording proxy between both libraries and one engine. Neither library reports
on itself:

```console
$ python3 scripts/compare_trajectories.py --engine http://127.0.0.1:8099/v1 --model gemma-4-e2b
recorded 2 asks from dsrust, 2 from dspy

Predict
-------
  message 0 (system): IDENTICAL, 876 bytes
  message 1 (user): IDENTICAL, 384 bytes
  every other request field: IDENTICAL ['model']

ChainOfThought
--------------
  message 0 (system): IDENTICAL, 934 bytes
  message 1 (user): IDENTICAL, 414 bytes
  every other request field: IDENTICAL ['model']

VERDICT: identical on every ask
```

Reproduce the lot:

```bash
git clone --recurse-submodules <repo>   # third_party/dspy at the pinned tag
cargo test --workspace                  # Rust suite + goldens, no Python
uv sync && ./scripts/run_upstream_tests.sh   # DSPy's own tests, over DsRust
```

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
- [x] **Custom-type seam**: `dspy.Type` + `Image` / `Audio` / `File` / `Code` / `Citations` /
  `Document` — and your own type on the same trait.

**Next**

- [ ] **The typed LM boundary, finished**: adapters answer with `LmMessage` rather than the older
  `ChatTurn`. The last slice of the DSPy 3.3 type migration, and the one breaking change left.

**Planned**

- [ ] **The rest of the sandbox.** DSPy's own interpreter suite runs against `DenoInterpreter`:
  37 of 44 pass, including host-callback tools, the typed `SUBMIT`, and file mounting with
  write-back. Two of the seven left are the >100MB variable path; the other five ask for a Python
  set, a nested tuple, or DSPy's own subprocess handle.
- [ ] Retrieval and the optimizers that need it (KNN, KNNFewShot, SIMBA); finetuning and RL.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

*DsRust is an independent port, not affiliated with or endorsed by the DSPy project or Stanford NLP.*
