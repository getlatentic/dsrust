# DsRust

A faithful, byte-for-byte Rust port of [DSPy](https://github.com/stanfordnlp/dspy): declare a task,
and the library writes the prompt, calls the model, and hands back typed values.

## Highlights

- **DSPy's prompts, not our own.** The bytes are identical, and DSPy's own pytest suite is what
  says so — it runs against DsRust's renderer. A
  [crossing counter](#how-the-fidelity-claim-is-tested) fails any test that passes without touching
  the Rust crate.
- **You declare a task, not a prompt.** `question -> answer` gives you the prompt, the call, the
  parsing, and the retry on both halves — a rate limit is asked again with DSPy's own backoff, and a
  reply that will not parse is re-asked through the JSON adapter. No template, no JSON handling.
- **Reuse a proven prompting technique** rather than reimplementing it: `ChainOfThought`, `ReAct`,
  `ReActV2`, `ProgramOfThought`, `CodeAct`, `RLM`, `BestOfN`, `Refine`.
- **Code-writing modules get a sandbox that ships.** `ProgramOfThought::new(sig)` runs the model's
  Python in DSPy's own `runner.js` under Deno and Pyodide — the same sandbox, defaulted the same
  way. `deno` is the one prerequisite, as it is for DSPy.
- **Optimize against your own metric.** MIPROv2 and GEPA, each reproduced down to its RNG and search
  order, so a compile here makes the choices a compile there makes.
- **Compiled programs cross the language line.** `dspy.load` opens what DsRust saves, and DsRust
  runs what Python saved. Same on-disk format, both directions.
- **Any OpenAI-compatible host is a base URL away**: OpenAI, Groq, Together, vLLM, LM Studio,
  `llama-server`. Anthropic and ollama have their own wires, and `ChatModel` is yours.
- **Watch a run two ways.** All six of DSPy's callback points — module, model call, tool, prompt
  rendering, reply parsing, evaluation — fire `Callback`, DSPy's `BaseCallback` as a trait with
  defaulted methods, *and* open a `tracing` span nested as the program is nested. Register handlers
  with `configure_callbacks`, or set `RUST_LOG=dsrust::observe=info` and write nothing at all.
- **Or skip the signature entirely.** `lm.call(items![User!["…"]])` is DSPy's `lm(dspy.User(...))`:
  a string, a conversation, or a previous reply handed straight back in for the next turn. The
  `User!`/`Assistant!`/`System!` macros take parts positionally the way DSPy's `*parts` does, so a
  string and an image sit in one turn — which an array cannot hold.
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

Those two are the whole list. `serde`, `serde_json` and `schemars` are not yours to add: the derive
expands through DsRust's own copies, and where the API takes a `serde_json::Value` — a signature
shaped at runtime, a tool's argument schema — `dsrust::serde_json::json!` builds one.

> **Status: alpha.** DSPy's own suite tests the prompt bytes, the parsing, the optimizers and the
> RNG. The API is smaller than DSPy's and grows toward it ([Roadmap](#roadmap)).

## A whole program

```rust
use dsrust::lm::{LM, configure};
use dsrust::{Predict, call};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure(LM::new("openai/gpt-4o-mini")?);   // reads OPENAI_API_KEY

    let qa = Predict!("question -> answer");
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

### Replies are cached, as DSPy caches them

An identical request is answered from disk rather than asked again — DSPy's `cache=True` default,
under `~/.dsrs_cache`. **Measuring a model means turning it off**, or a second run reads the first
run's answer and reports it as a fresh one:

```rust
LM::builder("openai/gpt-4o-mini").cache(false).build()?  // per model
```

```bash
DSRS_CACHEDIR=$(mktemp -d) cargo run                    # or a throwaway directory
```

## Documentation

[`docs/usage.md`](docs/usage.md) is the full guide: every module, every declaration spelling, and
the DSPy-vs-DsRust mapping side by side.

## Features

### A task is a string or a struct

```rust
let qa = Predict!("question -> answer");                       // names only
let m  = Predict!("question: str, context: list[str] -> answer: int");   // types too
```

```rust
#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]  question: String,
    #[output] answer: String,
}

let out = call!(Predict!(QA), question = "capital of France?").await?;
println!("{}", out.answer);   // a String, checked when this compiles
```

A bad signature fails the build. `Predict!("a -> b -> c")` is a compile error carrying DSPy's own
message, pointed at the literal; DSPy raises the equivalent when the program runs.

### Reaching a provider

The prefix is a **wire format**, not a brand, so a host is a base URL rather than a new provider.

```rust
configure(LM::new("openai/gpt-4o-mini")?);                  // OPENAI_API_KEY

configure(                                                   // Groq, same wire
    LM::builder("openai/llama-3.3-70b")
        .api_base("https://api.groq.com/openai/v1")
        .api_key(std::env::var("GROQ_API_KEY")?)
        .build()?,
);

configure(                                                   // a local llama-server
    LM::builder("openai/gemma-3-1b")
        .api_base("http://127.0.0.1:8080/v1")
        .api_key("not-needed-locally")
        .build()?,
);
```

`anthropic/…`, `ollama/…` and `ollama_chat/…` are the other built-in wires. Your own provider is the
`ChatModel` trait.

Settings that apply to every call go on the model, as DSPy's `dspy.LM(model, temperature=…)` do:

```rust
let lm = LM::builder("openai/gpt-4o-mini")   // the model is positional: it cannot be forgotten
    .temperature(0.5)
    .max_tokens(512)
    .build()?;
```

A single call still overrides them — the model's settings fill in what the call did not state, which
is DSPy's `{**lm.kwargs, **call_kwargs}`.

### A transient failure is asked again

DSPy's `num_retries=3` default, which counts asks rather than retries: a rate limit or a 5xx is
retried before you see it, backing off 1s then 2s and honouring `Retry-After` where the provider
sent one. Only the four kinds DSPy 3.3 names retryable are asked again, so a rejected key comes
straight back.

```rust
LM::builder("openai/gpt-4o-mini").num_retries(5).build()?   // or 1, to never ask twice
```

### Optimizing, and saving what it found

```rust
GEPA::new(metric, reflection_lm)   // evolves the prompt against your metric
    .max_metric_calls(200)
    .num_threads(8)                // most of a run is evaluation
    .compile(&mut program, &trainset, &valset)
    .await?;

program.save(std::path::Path::new("compiled.json"))?;   // DSPy's format; open it in Python
```

MIPROv2 searches instructions and few-shot demos together, at DSPy's own defaults of four
bootstrapped and four labelled — `max_bootstrapped_demos(0).max_labeled_demos(0)` is its zero-shot
run. It also runs to DSPy's budget presets: `auto(Auto::Light)` subsamples the validation set, sizes
both candidate counts and the trial count from it, and scores each trial on a minibatch with full
evaluations interleaved — the same search, not an approximation of it.

A compile rewrites the program in place, so what you save is the program you passed in. DSPy hands
back a new one instead; Rust has no deep copy of a `dyn Module` to hand back.

GEPA's strategy seams are DSPy's, as traits and enums rather than duck-typed callables:
`candidate_selection_strategy` (Pareto or current-best), `component_selector` (round-robin or all), and
`instruction_proposer` — a trait, so your own proposer can carry its own model and template, and it
replaces GEPA's reflection prompt entirely the way DSPy's `ProposalFn` does.

Run the whole loop — declare, ask, score, compile — with no provider and no API key:

```bash
cargo run --example quickstart
```

## How the fidelity claim is tested

Two layers, both against the pinned upstream (`dspy==3.3.0b1`, `gepa==0.1.1`, `optuna==4.9.0`,
`json-repair==0.61.7`), never against a transcription of it:

1. **Committed goldens** (`crates/dsrust/tests/conformance/**`), the exact bytes and decisions
   captured from *running* the pinned Python. `cargo test` checks against them, no Python needed.
2. **DSPy's own pytest suite, over DsRust.** A PyO3 bridge runs DSPy's actual tests with DsRust
   underneath. A crossing counter fails any test that passes *without* touching the Rust crate.

<!-- status: generated by scripts/status.py from backlog.toml [status] -->

Today: **1456 Rust tests pass**, and **1144 of DSPy's own**, across 58 of its 94 test files. 656 of those cross into the Rust crate.

<!-- /status -->

Plus byte-verified reproductions of CPython's Mersenne Twister (against CPython's *own*
`test_guaranteed_stable` vector), numpy's RNG, optuna's TPE sampler, the gepa engine, and
`json-repair` — the malformed-JSON reader `JSONAdapter.parse` opens with, which is a PyPI package
rather than DSPy's code and is not the crates.io crate of the same name.

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
  COPRO, MIPROv2 (instructions *and* few-shot demos), GEPA (reflective mutation *and* merge, with
  CPython set-order and RNG reproduced), BetterTogether.
- [x] **Save/load** in DSPy's on-disk format; `Reasoning`, `Tool`, `ToolCalls`, `History` + tool
  history; native function calling and native reasoning (`reasoning_effort`).
- [x] **A sandbox that ships**: `DenoInterpreter` runs DSPy's own `runner.js` under Deno and
  Pyodide, and DSPy's interpreter suite holds it — 39 of 44, with host-callback tools, the typed
  `SUBMIT`, file mounting with write-back and the >100MB variable path. The five left ask for a
  Python `set` as an input variable or for DSPy's own subprocess handle.
- [x] **Custom-type seam**: `dspy.Type` + `Image` / `Audio` / `File` / `Code` / `Citations` /
  `Document` — and your own type on the same trait.

**Next**

- [ ] **The typed LM boundary, finished**: adapters answer with `LmMessage` rather than the older
  `ChatTurn`. The last slice of the DSPy 3.3 type migration, and the one breaking change left.
- [ ] **Drop the HTTP client from `ChatModel`**: DSPy's `forward(request)` names none, and ours
  makes a caller's own provider depend on `reqwest` at a matched version.

**Planned**

- [ ] Retrieval and the optimizers that need it (KNN, KNNFewShot, SIMBA); finetuning and RL.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

*DsRust is an independent port, not affiliated with or endorsed by the DSPy project or Stanford NLP.*
