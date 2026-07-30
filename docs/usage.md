# Declaring a task, and asking it

Every Rust shape here is compiled and run by [`crates/dsrust/tests/every_spelling.rs`](../crates/dsrust/tests/every_spelling.rs).
If one of them stops being true, that test fails rather than this page quietly going stale.

A task is declared one of two ways — its field names in a string, or a struct — and asked by one
of two modules. Both declarations produce the same type and are asked the same way, which is what
dspy gives by having one `Predict` class and one `Signature` base.

## What a program needs

```bash
cargo add dsrust@0.1.0-alpha.2
cargo add tokio --features macros,rt-multi-thread
```

Asking a model is a network call, so every call is `async` and the program needs a runtime. Tokio
is the only crate you add beside this one. The flavour is yours: DsRust names no tokio type, so it
does not choose a scheduler on your behalf. `rt-multi-thread` is what `#[tokio::main]` uses without
arguments; a CLI wanting one thread asks for `rt` and `#[tokio::main(flavor = "current_thread")]`.

Every snippet below is a fragment. Here is one inside a whole program:

```rust
use dsrust::lm::{LM, configure};
use dsrust::{Signature, call, predict};

#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure(LM::new("openai/gpt-4o-mini")?);   // reads OPENAI_API_KEY

    let out = call!(predict!(QA), question = "What is the capital of France?").await?;
    println!("{}", out.answer);
    Ok(())
}
```

`out.answer` is a `String` because the task was declared as a struct. A task declared as a string
has no struct to fill. Its fields arrive as JSON, so `out.get("answer")` answers with a `&Value`.
Call `as_str()` on it to print `Paris` rather than `"Paris"`.

[`scripts/check_external_consumer.sh`](../scripts/check_external_consumer.sh) compiles that program
on every gate run, from a crate outside this repository. So the dependency list above is the one
that works, not the one someone remembered.

`configure` sets the model for the whole process. Pass one to a single module instead with
`Predict::with_lm`, which is what an optimizer varies; see
[Reaching a provider](#reaching-a-provider-and-adding-your-own).

## The shape of it

```rust
let qa = predict!("question -> answer");                   // declare
let out = call!(qa, question = "capital of France?").await?;   // ask
out.get("answer").unwrap()                                  // read
```

Three things are worth knowing before the table.

**Building is free; asking is the network call.** `predict!` reaches no provider, and neither does
copying the module. The model is reached in `call!`, once per call, which is why that line and only
that line carries `.await?`.

**A bad signature fails the build.** `predict!("a -> b -> c")` is a compile error carrying dspy's
own message, pointed at the literal. dspy raises the equivalent when the program runs.

**A module is a value.** That is what lets an optimizer rewrite it, and the reason a callable was
not the right shape:

```rust
let mut program = predict!("question -> answer");
BootstrapFewShot::new(exact_match).compile(&mut program, &trainset).await?;
let out = call!(program, question = "…").await?;   // the same line, now compiled
```

## Field names in a string

### One in, one out

```python
qa = dspy.Predict("question -> answer")
out = qa(question="capital of France?")
out.answer
```

```rust
// the whole way
let signature: Signature = "question -> answer".parse()?;
let qa = Predict::from_signature(signature);
let out = qa.forward(input! { question: "capital of France?" }).await?;
out.get("answer").unwrap()

// the short way
let qa = predict!("question -> answer");
let out = call!(qa, question = "capital of France?").await?;
```

### Two in, two out

```python
qa = dspy.Predict("subject, tone -> haiku, mood")
out = qa(subject="computer science", tone="wry")
```

```rust
let qa = predict!("subject, tone -> haiku, mood");
let out = call!(qa, subject = "computer science", tone = "wry").await?;
out.get("haiku").unwrap()
```

Types are allowed, and a comma inside brackets belongs to the type rather than separating a field:

```rust
let m = predict!("question: str, context: list[str] -> answer: int, scores: dict[str, float]");
```

## A struct

```python
class QA(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

```rust
#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}
```

### One in, one out

```python
qa = dspy.Predict(QA)
out = qa(question="capital of France?")
out.answer
```

```rust
// the whole way — the task's own inputs struct in, its own outputs struct back
let qa = Predict::<QA>::new();
let out = qa.call_inputs(&QAInputs { question: "capital of France?".into() }).await?;
out.answer

// the short way — the same spelling a string signature is asked with
let qa = predict!(QA);
let out = call!(qa, question = "capital of France?").await?;
out.answer
```

`out.answer` rather than a lookup: a derived task knows its outputs, so the field is checked when
this compiles. That is the reason the two declarations answer with different types even though
they are built and asked alike.

### Two in, two out

```python
poet = dspy.Predict(Haiku)
out = poet(subject="quantum computing", tone="wry")
```

```rust
let poet = predict!(Haiku);
let out = call!(poet, subject = "quantum computing", tone = "wry").await?;
out.haiku
```

One invocation can name the task and fill it, evaluating to the call itself. The inputs literal is
exhaustive, so a forgotten field is a compile error:

```rust
let out = predict!(Haiku {
    subject: "quantum computing",
    tone: "wry"
})
.await?;
```

## Chain of thought

The model is asked for a leading `reasoning` field, which is kept out of the answer. Everything
else reads the same.

```python
c = dspy.ChainOfThought("question -> answer")
out = c(question="a calm colour?")

c = dspy.ChainOfThought(Haiku)
out = c(subject="machine learning", tone="patient")
```

```rust
let picked = chain_of_thought!("question -> answer");
let out = call!(picked, question = "a calm colour?").await?;

let out = chain_of_thought!(Haiku {
    subject: "machine learning",
    tone: "patient"
})
.await?;
out.haiku
```

## A module of your own

Composing steps into one program is the point of the `Module` seam, and it is what an optimizer
walks. Python subclasses; Rust implements a trait.

```python
class Outline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict("subject -> angle")
        self.write = dspy.Predict("angle -> haiku")

    def forward(self, subject):
        angle = self.plan(subject=subject).angle
        return self.write(angle=angle)
```

```rust
#[derive(Module)]
struct Outline {
    plan: Predict,
    write: Predict,
}

impl Forward for Outline {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let angle = self.plan.forward(inputs).await?;
        let handed = input! { angle: angle.get("angle").cloned().unwrap_or_default() };
        self.write.forward(handed).await
    }
}
```

```rust
let mut mine = Outline::new();
let out = call!(mine, subject = "winter mornings").await?;
```

You write how it runs; the derive writes what Python inherits.

Every named field is a step. `named_predictors` — the seam an optimizer works through — comes from
that field list. The derive renames each child's predictors after the field holding them, so a demo
says which step earned it. A field that is not a step carries `#[not_a_step]`. The derive also makes
the module callable through `call!`.

`Forward` exists so an author is not writing `Pin<Box<dyn Future>>` by hand. `Module` keeps that
shape because it must be object-safe for a composed program to hold `Box<dyn Module>`; the derive
does the boxing in between.

## Reaching a provider, and adding your own

A model is an `LM`, named `provider/model-id`. The prefix picks a **wire format**, not a brand.
`openai/…` is the OpenAI `/v1/chat/completions` shape. OpenAI, Groq, Together, Fireworks, DeepSeek,
vLLM, LM Studio and `llama-server` all speak it. You reach any of them by pointing the base URL, not
by adding a prefix. `anthropic/…` and `ollama/…` are their own shapes.

```rust
// OpenAI itself, from OPENAI_API_KEY in the environment.
let lm = LM::new("openai/gpt-4o-mini")?;

// Groq, on the same wire, a different host and key.
let lm = LM::new("openai/llama-3.3-70b")?
    .with_openai_base_url("https://api.groq.com/openai/v1")
    .with_openai_key(std::env::var("GROQ_API_KEY")?);

dsrust::lm::configure(lm); // the process-wide default every module reaches
```

This is where DsRust and dspy part on purpose. dspy routes every provider through **litellm**, where
the prefix is the brand (`groq/…`, `bedrock/…`) and litellm knows the host. DsRust carries no
litellm. The prefix is the wire format, and you name a non-OpenAI host by its URL. The four built-in
shapes are a closed `match`, because *something* must map a model string to a wire. dspy has the
same map inside litellm, just hidden.

**A provider of your own is the `ChatModel` trait**, the one seam every built-in already implements:

```rust
struct MyProvider {
    key: String,
}

impl ChatModel for MyProvider {
    fn forward<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a dsrust::lm::api::LmRequest,
    ) -> impl Future<Output = Result<dsrust::lm::api::LmResponse>> + Send + 'a {
        async move {
            // request.wire_messages(), request.config.temperature, request.output_schema() —
            // translate the typed request to your API, and its reply back to an LmResponse.
            todo!()
        }
    }
}
```

This is dspy 3.3's `forward_contract = "typed_lm"` shape exactly: `forward(LmRequest) -> LmResponse`.
A model built this way is indistinguishable from the built-ins — it nests behind `Cached`, reaches
every module, and (with `forward_stream`, optional) streams.

Rust has no class inheritance, so "extend OpenAI and change one thing" is **composition**: the
built-in OpenAI provider is `Endpoint`, parameterised by base URL, key, JSON envelope and
token-cap rule. A variant that differs only in configuration is therefore a different `Endpoint`,
not a subclass. A provider that shares the OpenAI wire but changes a header or the reply parsing
wraps the OpenAI request and reply pieces in its own `ChatModel`. It holds what it reuses rather
than inheriting it.

### Settings that apply to every call

DSPy keeps `temperature` and `max_tokens` on the LM and merges them beneath each call —
`kwargs = {**self.kwargs, **kwargs}`. So does this:

```rust
configure(
    LM::builder("openai/gpt-4o-mini")   // the model is positional: it cannot be forgotten
        .temperature(0.2)
        .max_tokens(512)
        .build()?,
);
```

A single call still overrides them; the model's settings fill in only what the call left unset.

### Replies are cached

An identical request is answered from disk rather than asked again — DSPy's `cache=True` default,
under `~/.dsrs_cache`. **Measuring a model means turning it off**, or a second run reads the first
run's answer and reports it as fresh:

```rust
LM::builder("openai/gpt-4o-mini").cache(false).build()?
```

```bash
DSRS_CACHEDIR=$(mktemp -d) cargo run    # or a throwaway directory
```

### When a call fails

Provider failures arrive as a typed `LmFailure`, DSPy 3.3's normalized LM errors:

```rust
match extractor.forward(inputs).await {
    Ok(out) => …,
    Err(error) => match error.downcast_ref::<LmFailure>() {
        // rate limit, timeout, server, transport — and honour `retry_after` when it is set
        Some(failed) if failed.is_retryable() => back_off(failed.retry_after).await,
        Some(failed) => eprintln!("{}: {}", failed.kind, failed.message),
        // Not a provider failure: a reply that would not parse or coerce.
        None => eprintln!("{error:#}"),
    },
}
```

`{error:#}` and not `{error}`: a parse or coercion failure keeps its cause in the chain, so the
short form names the category and the alternate form names the field.

A reply that parses but is missing a field is re-asked through the JSON adapter, as DSPy does it.
`Predict::with_feedback_retry` swaps that for a second ask in the original format, carrying the
error. That recovery is this crate's own and DSPy has no equivalent, so it is off by default.

## Every module, and what it takes

dspy's modules split into two families, and the split decides how you build one.

**A module that takes a signature** is declared like `Predict`: a field-name string, or a task
type. Each has a macro of the same name.

**A wrapper takes another module.** There is no signature to hand it; the signature lives in
whatever it wraps, so it is built with `::new` and has no macro.

| module | takes | dspy | DsRust |
|---|---|---|---|
| `Predict` | a signature | `dspy.Predict("q -> a")` | `predict!("q -> a")` |
| `ChainOfThought` | a signature | `dspy.ChainOfThought("q -> a")` | `chain_of_thought!("q -> a")` |
| `ReAct` | a signature + tools | `dspy.ReAct("q -> a", tools=[…])` | `react!("q -> a", tools)` |
| `ReActV2` | a signature + tools | `dspy.ReActV2("q -> a", tools=[…])` | `react_v2!("q -> a", tools)` |
| `ProgramOfThought` | a signature | `dspy.ProgramOfThought("q -> a")` | `program_of_thought!("q -> a")` |
| `CodeAct` | a signature + tools | `dspy.CodeAct("q -> a", tools=[…])` | `code_act!("q -> a", tools)` |
| `RLM` | a signature | `dspy.RLM("q -> a")` | `rlm!("q -> a")` |
| `MultiChainComparison` | a signature | `dspy.MultiChainComparison("q -> a")` | `MultiChainComparison::with_attempts(…)` |
| `BestOfN` | **a module** | `dspy.BestOfN(module=qa, N=3, reward_fn=f, threshold=1.0)` | `best_of_n!(qa, n = 3, reward = f, threshold = 1.0)` |
| `Refine` | **a module** | `dspy.Refine(module=qa, N=3, …)` | `refine!(qa, n = 3, reward = f, threshold = 1.0)` |
| `Parallel` | branches per call | `dspy.Parallel(num_threads=8)` | `Parallel::new(8)` |

Every signature-taking macro accepts **either spelling** — a string, or a task declared with
`#[derive(Signature)]`:

```rust
let quick   = predict!("question -> answer");
let declared = predict!(Investigate);          // carries its doc comment as instructions
let agent    = react!(Investigate, tools, max_iters = 4);
let reader   = rlm!(Investigate, max_iterations = 6);
```

Each keeps the cap keyword its own module uses: `max_iters` for most, `max_iterations` for `RLM`.
A uniform name would be one none of them actually have.

### Code-writing modules run real Python

`ProgramOfThought`, `CodeAct` and `RLM` default to a Deno/Pyodide sandbox running DSPy's own
`runner.js`, exactly as `dspy.ProgramOfThought(...)` defaults to `PythonInterpreter()`. **`deno`
must be on the path**, which is what DSPy asks of its users too:

```bash
curl -fsSL https://deno.land/install.sh | sh    # or: brew install deno
```

Supply your own environment with `with_interpreter`, which is also how a test scripts one without
needing deno at all.

### Asking one, side by side

```python
# dspy
qa = dspy.ChainOfThought("question -> answer")
out = qa(question="capital of France?")
out.answer
```

```rust
// DsRust
let qa = chain_of_thought!("question -> answer");
let out = call!(qa, question = "capital of France?").await?;
out.get("answer").unwrap()
```

### Wrapping one

`BestOfN` runs a module up to `n` times, each at a fresh rollout and `temperature = 1.0`, and
keeps the highest-scoring attempt — stopping early at the first to reach `threshold`.

```python
# dspy
def one_word(inputs, out):
    return 1.0 if len(out.answer.split()) == 1 else 0.0

qa = dspy.ChainOfThought("question -> answer")
best = dspy.BestOfN(module=qa, N=3, reward_fn=one_word, threshold=1.0)
out = best(question="capital of Belgium?")
```

```rust
// DsRust
fn one_word(_inputs: &Example, out: &Prediction) -> f64 {
    match out.get("answer").and_then(|answer| answer.as_str()) {
        Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
        _ => 0.0,
    }
}

let qa = chain_of_thought!("question -> answer");
let best = best_of_n!(qa, n = 3, reward = one_word, threshold = 1.0);
let out = call!(best, question = "capital of Belgium?").await?;
```

The reward is a named function in both, which is what dspy's own example does. A Rust closure
works too — `|inputs: &Example, out: &Prediction| …` — but writing one inline inside the
constructor buries the three arguments around it.

**`best_of_n!` names the arguments** because dspy passes all four by keyword and Rust has no named
arguments. `BestOfN::new(qa, 3, one_word, 1.0)` compiles and says nothing about which number is
`n` and which is `threshold`. This is the same reason `call!` and `input!` exist: the macros
supply what the language does not. `fail_count` is optional in the macro as it is upstream.

All four arguments are upstream's. Two of its details are easy to read as bugs and are not:
`threshold` is **required**, because `BestOfN.forward` compares against it with no guard; and
`fail_count = 0` means *n*, not *none allowed*, because dspy reads `fail_count or N` and Python
treats zero as unset.

A wrapper is still a module — it nests, `call!` reaches it, and an optimizer's walk goes straight
through to the predictors inside it.

## Against dspy

| | dspy | DsRust |
|---|---|---|
| Constructor | `dspy.Predict(x)` | `predict!(x)` — a string or a task |
| Call | `m(field=…)` | `call!(m, field = …)` |
| Reading a result | `out.answer` | `out.answer` typed, `out.get("answer")` from a string signature |
| One module type | ✅ | ✅ `Predict<S = Dynamic>` |
| Optimizable either way | ✅ | ✅ |
| A malformed signature | raises when run | fails the build |

Three differences are the language rather than the design:

- **`call!(m, …)` puts the module inside the parentheses.** Postfix macros are not Rust syntax, so
  `m.call!(…)` cannot parse. `m.forward(input! { … })` keeps the module as the subject if that
  reads better.
- **`.await?` is on the call.** It is the network request and its failure, and it appears only
  where the model is actually reached.
- **The typed long form is `call_inputs`, not `call`.** Rust cannot resolve two methods of one
  name across `Predict<Dynamic>` and `Predict<Task>`, so the two constructors and the two typed
  calls carry distinct names. `call!` is unaffected and works on both.

## Why macros

Rust has no mapping literal and no named arguments, so `{ subject: "…" }` and `f(subject = "…")`
are both syntax errors. `input!` and `call!` supply what the language does not, the way `vec!`
supplies a list literal. `predict!` additionally checks a signature string while the crate
compiles, which is something dspy cannot do at all.
