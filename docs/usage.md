# Declaring a task, and asking it

Every Rust block on this page is extracted and compiled by
[`scripts/check_docs.py`](../scripts/check_docs.py), from a crate outside this repository with only
the two dependencies named below. [`crates/dsrust/tests/every_spelling.rs`](../crates/dsrust/tests/every_spelling.rs)
then *runs* the shapes against a scripted model. So a page that goes stale fails a gate.

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
use dsrust::{Predict, Signature, call};

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

    let out = call!(Predict!(QA), question = "What is the capital of France?").await?;
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
`Predict::set_lm`, which is what an optimizer varies; see
[Reaching a provider](#reaching-a-provider-and-adding-your-own).

To point a program you already have at another model for one piece of work — DSPy's
`with dspy.context(lm=...)` — scope it:

```rust
use dsrust::lm::context;

let answered = context(LM::new("anthropic/claude-sonnet-4-5")?)
    .run(program.forward(inputs))
    .await?;
```

Every module the work reaches asks the scoped model, however deeply nested and without any of them
being rebuilt. That is the difference from `set_lm`, which is a construction-time choice on one
module: a five-module pipeline someone handed you can be redirected this way and no other.

It scopes a *future* rather than a block. DSPy can use a `with` statement because a `ContextVar` in
asyncio is per-Task; a Rust guard held across an `.await` would instead be read by whatever the
runtime polled next. `run` enters on each poll, so two pieces of work interleaved in one task each
see their own model.

## The shape of it

```rust
let qa = Predict!("question -> answer");                   // declare
let out = call!(qa, question = "capital of France?").await?;   // ask
out.get("answer").unwrap()                                  // read
```

Three things are worth knowing before the table.

**Building is free; asking is the network call.** `Predict!` reaches no provider, and neither does
copying the module. The model is reached in `call!`, once per call, which is why that line and only
that line carries `.await?`.

**A bad signature fails the build.** `Predict!("a -> b -> c")` is a compile error carrying dspy's
own message, pointed at the literal. dspy raises the equivalent when the program runs.

**A module is a value.** That is what lets an optimizer rewrite it, and the reason a callable was
not the right shape:

```rust
let mut program = Predict!("question -> answer");
BootstrapFewShot::new(exact_match).compile(&mut program, &trainset).await?;
let out = call!(program, question = "…").await?;   // the same line, now compiled
```

A trainset is a list of `Example`s, and `example!` is dspy's `dspy.Example(question=…, answer=…)` —
a macro because Python takes keyword arguments and Rust does not. `with_inputs` says which fields
the model is given; the rest are what it is scored against:

```rust
let trainset = vec![
    example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
    example! { question: "capital of Japan?", answer: "Tokyo" }.with_inputs(["question"]),
];
```

## Naming one of DSPy's own types in a string

A signature string names Python's types, and dspy resolves the annotation against builtins and
`typing` — plus `dspy` itself, and nothing else. So its own types are reached **through the
module**:

```rust
let transcribe = Predict!("clip: dspy.Audio -> transcript");
let chat = Predict!("question: str, history: dspy.History -> answer");
```

The bare name is not a name upstream knows: `dspy.Signature("q -> a: History")` raises
`ValueError: Unknown name: History`, and so does this. The dotted spelling is what makes the same
string mean the same thing in both languages — which is the point of a signature being a string.

A type of your own is not reachable this way in either language. Declare it with
`#[derive(Signature)]`, where the Rust type is named directly.

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
let answer = out.get("answer").unwrap();

// the short way
let qa = Predict!("question -> answer");
let out = call!(qa, question = "capital of France?").await?;
```

`parse()` reads the string when the program runs, so a typo in it is a runtime error — which is
what dspy does. Where the string is written in your source, `make_signature!` reads it while the
crate compiles instead, and a bad one fails the build:

```rust
let signature = make_signature!("question -> answer");
```

That is what `Predict!("question -> answer")` uses internally. Reach for it directly when you need
the `Signature` itself rather than a module around it — `ReActV2::new` takes one, where
`ReActV2!` takes the literal.

### Two in, two out

```python
qa = dspy.Predict("subject, tone -> haiku, mood")
out = qa(subject="computer science", tone="wry")
```

```rust
let qa = Predict!("subject, tone -> haiku, mood");
let out = call!(qa, subject = "computer science", tone = "wry").await?;
out.get("haiku").unwrap()
```

Types are allowed, and a comma inside brackets belongs to the type rather than separating a field:

```rust
let m = Predict!("question: str, context: list[str] -> answer: int, scores: dict[str, float]");
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
let answer = out.answer;

// the short way — the same spelling a string signature is asked with
let qa = Predict!(QA);
let out = call!(qa, question = "capital of France?").await?;
let answer = out.answer;
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
let poet = Predict!(Haiku);
let out = call!(poet, subject = "quantum computing", tone = "wry").await?;
out.haiku
```

One invocation can name the task and fill it, evaluating to the call itself. The inputs literal is
exhaustive, so a forgotten field is a compile error:

```rust
let out = Predict!(Haiku {
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
let picked = ChainOfThought!("question -> answer");
let out = call!(picked, question = "a calm colour?").await?;

let out = ChainOfThought!(Haiku {
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

impl Outline {
    fn new() -> Self {
        Self {
            plan: Predict!("subject -> angle"),
            write: Predict!("angle -> haiku"),
        }
    }
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

`call!` reaches it because the derive writes the `Ask` impl — the trait `call!` goes through, whose
`Answer` is what decides how the result reads. Left alone that is a `Prediction`, so `out.get(..)`;
name the task the program answers with and it becomes that task's own struct:

```rust
#[derive(Module)]
#[task(Haiku)]
struct Outlined {
    plan: Predict,
    write: Predict,
}

impl Forward for Outlined {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        self.write.forward(self.plan.forward(inputs).await?.example).await
    }
}
```

Then `call!(mine, subject = "winter mornings").await?.haiku` is checked when it compiles, the same
line `Predict!(Haiku)` and `ReActV2!(Haiku, tools)` answer with. Leave it off for a program whose
outputs are not one task's.

`Forward` exists so an author is not writing `Pin<Box<dyn Future>>` by hand. `Module` keeps that
shape because it must be object-safe for a composed program to hold `Box<dyn Module>`; the derive
does the boxing in between.

**Use the derive.** Implementing `Module` by hand compiles, and quietly costs two things the
derive writes: `named_predictors` takes the trait's empty default, so an optimizer walks no
predictors, rewrites nothing and reports success; and no point is opened, so the module is
invisible to every callback and shows in a trace as its children with nothing above them. Measured,
one call each:

| | `on_module_start` | `named_predictors` |
|---|---|---|
| `#[derive(Module)]` + `Forward` | fires, named after the struct | 2 |
| hand-written `impl Module` | **silent** | **0** |

Neither failure says anything at the time, which is why this is a recommendation rather than a
choice. Write the impl by hand only for a module with no predictors under it and nothing to watch —
otherwise write those two members as well.

## Reaching a provider, and adding your own

A model is an `LM`, named `provider/model-id`. The prefix picks a **wire format**, not a brand.
`openai/…` is the OpenAI `/v1/chat/completions` shape. OpenAI, Groq, Together, Fireworks, DeepSeek,
vLLM, LM Studio and `llama-server` all speak it. You reach any of them by pointing the base URL, not
by adding a prefix. `anthropic/…` and `ollama/…` are their own shapes.

```rust
// OpenAI itself, from OPENAI_API_KEY in the environment.
let lm = LM::builder("openai/gpt-4o-mini").build()?;

// Groq, on the same wire, a different host and key.
let lm = LM::builder("openai/llama-3.3-70b")
    .api_base("https://api.groq.com/openai/v1")
    .api_key(std::env::var("GROQ_API_KEY")?)
    .build()?;

// Anthropic. `api_key` follows the model's prefix, so this reaches ANTHROPIC's field.
let lm = LM::builder("anthropic/claude-sonnet-4-5")
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .build()?;

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
One request in, one response out, and nothing else — no HTTP client, as DSPy's names none. Your own
provider brings whatever transport it likes; the built-ins share one, which
`configure_with_client` is how you replace.
A model built this way is indistinguishable from the built-ins — it nests behind `Cached`, reaches
every module, and (with `forward_stream`, optional) streams.

Rust has no class inheritance, so "extend OpenAI and change one thing" is **composition**: the
built-in OpenAI provider is `Endpoint`, parameterised by base URL, key, JSON envelope and
token-cap rule. A variant that differs only in configuration is therefore a different `Endpoint`,
not a subclass. A provider that shares the OpenAI wire but changes a header or the reply parsing
wraps the OpenAI request and reply pieces in its own `ChatModel`. It holds what it reuses rather
than inheriting it.

### Asking a model directly, with no signature

DSPy's `lm(...)` — its `BaseLM.__call__` — is `ChatModel::call`. Every model has it, including one of
your own, because it is defaulted on the trait and built from `forward`.

```python
lm = dspy.LM("openai/gpt-4o-mini")
response = lm(dspy.User("What is DSPy?"))
```

```rust
let lm = LM::new("openai/gpt-4o-mini")?;

let answered = lm.call(items![User!["What is the capital of France?"]]).await?;

// A reply goes straight back in as the assistant turn it was.
let next = lm.call(items![answered, User!["And of Belgium?"]]).await?;

// Prose and an image are one multimodal turn, not two.
lm.call(items![User!["Describe this.", LmPart::image_url(url)]]).await?;

// A named speaker, which is how two `user` turns stay apart.
lm.call(items![User!["hello"].name("alice"), Assistant!["hi"].name("bot")]).await?;
```

`User`, `Assistant`, `System` and `Developer` are DSPy's own names for the four role constructors —
free functions there carrying `# noqa: N802`, and here carrying `#[allow(non_snake_case)]`, the same
trade `Predict!` makes. `LmMessage::user(…)` is the same constructor under Rust's conventions.

Each has a macro beside it, and the macro is the one to reach for. DSPy's constructors take
`*parts`, so `dspy.User("Describe this:", image)` writes two parts of different types positionally.
A Rust function cannot: `User(["Describe this:", image])` is an array, and every element of an array
is one type. `User!["Describe this:", image]` converts each expression on its own, so it can. The
macros stay thin — each expands to the function beside it, so nothing this crate decides lives
inside one.

An image in a turn is `LmPart::image_url(url)`, DSPy's `LMImagePart`. It is *not* `Image`, which is
the type a declared signature field carries — a different layer. DSPy 3.3.0b1's own `User` docstring
shows `dspy.Image(...)` in this position and that raises; upstream's main has since rewritten those
examples to `LMImagePart(url=…)`, so this spelling is the corrected one rather than a divergence.

A speaker's name is `.name("alice")`, DSPy's `name=` keyword. It reaches the wire — OpenAI takes
`messages[].name` right after the role — which is how a multi-agent transcript keeps two `user`
turns apart. `.metadata(…)` is DSPy's other keyword and stays runtime-only, so it is a place for a
trace id rather than a way to reach the provider.

`items!` is the same idea one level up, for the conversation rather than the turn: a run mixing a
turn, a reply and a string needs each element converted — the same reason `call!` and `input!`
exist.

Two doors, one request. A signature renders to messages and enters through
`LmRequest::from_messages`; a direct call enters through `LmRequest::from_items`. Below that they are
the same type, which is why a module needs no special handling for either. DSPy has one constructor
that raises when you pass both; two constructors make that unwritable.

`lm(...)` itself is not possible on stable Rust — calling a struct needs `fn_traits` and the
`rust-call` ABI, both unstable — so the method carries DSPy's own name for it.

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

### A transient failure is asked again

A rate limit or a 5xx is retried before you see it — DSPy's `LM(num_retries=3)`, which counts *asks*
rather than retries, so the default is three asks and two retries. A rate limit backs off 1s then 2s
(and honours `Retry-After` when the provider sends one); anything else is asked again immediately.

```rust
LM::builder("openai/gpt-4o-mini").num_retries(5).build()?   // or 1, to never ask twice
```

Only the four kinds DSPy 3.3 calls retryable are asked again — rate limit, timeout, server,
transport. A rejected key or a malformed request fails the same way twice, so it comes straight back.

### The o1 family's `developer` role

Those models take `developer` where everything else takes `system`. DSPy's
`LM(use_developer_role=True)` renames the system message on the way out, and so does this — on the
Responses wire only, as upstream has it:

```rust
LM::builder("openai/o1-mini")
    .use_developer_role(true)
    .build()?
```

Writing the turn yourself, `Developer(["…"])` is the constructor.

### Watching a run

DSPy fires a `BaseCallback` at six points — a module's run, a model call, a tool call, rendering the
prompt, reading the reply back, and an evaluation — each with a start and an end. All six are here,
and each fires two things: your `Callback` handlers, and a `tracing` span. Use either.

#### As callbacks

`BaseCallback` is a base class whose methods are no-ops, which is a trait with defaulted methods.
Implement the handlers you care about and leave the rest:

```rust
use dsrust::{CallId, Callback, configure_callbacks, observe};

struct Logging;

impl Callback for Logging {
    fn on_module_start(&self, call: &CallId, module: &str, inputs: &Example) {
        println!("[{call}] {module} asked {}", observe::as_json(inputs));
    }

    fn on_module_end(&self, call: &CallId, answered: Result<&Prediction, &anyhow::Error>) {
        match answered {
            Ok(prediction) => println!("[{call}] -> {}", observe::as_json(&prediction.example)),
            Err(error) => println!("[{call}] failed: {error:#}"),
        }
    }
}
```

```rust
configure_callbacks([Arc::new(Logging) as Arc<dyn Callback>]);
```

That is DSPy's `dspy.configure(callbacks=[LoggingCallback()])`. To watch one model rather than the
whole process — DSPy's `dspy.LM(model, callbacks=[…])` — put the list on the model instead:

```rust
let lm = LM::builder("openai/gpt-4o-mini")
    .callbacks([Arc::new(Logging) as Arc<dyn Callback>])
    .build()?;
```

Both lists are told, the process-wide ones first.

A `CallId` is the same value at a point's start and its end, and `call.parent()` is the call it
happened inside — so `ChainOfThought` encloses its `Predict`, which encloses the model call. That is
DSPy's `call_id` and `ACTIVE_CALL_ID`, without the second lookup. A handler that panics is caught and
logged rather than allowed to end the run, as DSPy wraps each of its own in `try/except`.

#### As spans

The same six points open a `tracing` span, which is the shape a Rust program already collects. A
composed program nests, so the span tree is the program's shape and an `lm` span sits inside
whichever module made the call. Each carries the values DSPy hands its handlers: the inputs on the
way in, and either the outputs or the failure on the way out.

```bash
cargo add tracing-subscriber --features env-filter
```

```rust
tracing_subscriber::fmt()
    .with_env_filter("dsrust::observe=info")
    .init();
```

```bash
RUST_LOG=dsrust::observe=info cargo run
```

An agent's tool calls are spans too, nested inside the module that made them — the arguments on the
way in, and either what the tool returned or why it refused on the way out. That is what DSPy's own
documentation example prints.

Rendering the prompt and reading the reply back are their own spans too, so a parse failure shows
the raw text beside the error — which is nearly always the next question.

An `Evaluate` run is one span with every row nested inside it, so filtering to `evaluate` gives one
line per scoring pass rather than one per row — which is what makes an optimizer's search readable.

Nothing is rendered when nothing is listening, so a program with neither a subscriber nor a
registered handler pays an atomic load per point.

One thing `tracing` asks of you that DSPy does not: a module you put behind `tokio::spawn` starts a
new span tree unless you carry the current one into it with `.in_current_span()`. Nothing inside this
crate crosses a thread, so its own nesting is intact either way.

Reach for a span over a callback when you want the two properties it has and a handler list does not:
a subscriber cannot change what it is shown, and a broken one cannot break the run. Upstream's own
documentation warns readers against mutating what a callback is handed, which is the same worry.

### When a call fails

What survives the retry arrives as a typed `LmFailure`, DSPy 3.3's normalized LM errors:

```rust
match extractor.forward(inputs).await {
    Ok(out) => …,
    Err(error) => match error.downcast_ref::<LmFailure>() {
        // Retryable and still here, so the budget ran out. Waiting longer is the caller's call.
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
`Predict::feedback_retry` swaps that for a second ask in the original format, carrying the
error. That recovery is this crate's own and DSPy has no equivalent, so it is off by default.

## Every module, and what it takes

dspy's modules split into two families, and the split decides how you build one.

**A module that takes a signature** is declared like `Predict`: a field-name string, or a task
type. Each has a macro of the same name.

**A wrapper takes another module.** There is no signature to hand it; the signature lives in
whatever it wraps, so it is built with `::new` and has no macro.

| module | takes | dspy | DsRust |
|---|---|---|---|
| `Predict` | a signature | `dspy.Predict("q -> a")` | `Predict!("q -> a")` |
| `ChainOfThought` | a signature | `dspy.ChainOfThought("q -> a")` | `ChainOfThought!("q -> a")` |
| `ReAct` | a signature + tools | `dspy.ReAct("q -> a", tools=[…])` | `ReAct!("q -> a", tools)` |
| `ReActV2` | a signature + tools | `dspy.ReActV2("q -> a", tools=[…])` | `ReActV2!("q -> a", tools)` |
| `ProgramOfThought` | a signature | `dspy.ProgramOfThought("q -> a")` | `ProgramOfThought!("q -> a")` |
| `CodeAct` | a signature + tools | `dspy.CodeAct("q -> a", tools=[…])` | `CodeAct!("q -> a", tools)` |
| `RLM` | a signature | `dspy.RLM("q -> a")` | `RLM!("q -> a")` |
| `MultiChainComparison` | a signature | `dspy.MultiChainComparison("q -> a")` | `MultiChainComparison::attempts(…)` |
| `BestOfN` | **a module** | `dspy.BestOfN(module=qa, N=3, reward_fn=f, threshold=1.0)` | `BestOfN!(qa, n = 3, reward = f, threshold = 1.0)` |
| `Refine` | **a module** | `dspy.Refine(module=qa, N=3, …)` | `Refine!(qa, n = 3, reward = f, threshold = 1.0)` |
| `Parallel` | branches per call | `dspy.Parallel(num_threads=8)` | `Parallel::new(8)` |

Scoring a devset takes the same knob: `Evaluate::new(devset, program, metric).num_threads(8)` is
DSPy's `Evaluate(num_threads=8)`, and rows come back in devset order however many ran at once.

A run also gives up once ten rows have failed — DSPy's `max_errors`, from its `settings.max_errors`.
A failing row still scores `failure_score` rather than aborting, but a devset run against a provider
that is simply down should say so instead of reporting a confident zero.

```rust
Evaluate::new(devset, |inputs| program.forward(inputs), exact_match)
    .num_threads(8)
    .max_errors(25)
```

The program is a closure rather than the module itself, because what is scored need not be a
`Module` at all — a metric run over anything that answers an `Example` is still an evaluation.

### A tool is a function

dspy reads a tool off a plain callable: the docstring becomes the description the model is shown,
and the type hints become the argument schema. There is no decorator — `dspy.Tool(fn)` inspects the
function it is handed, at run time. Rust erases a doc comment long before the program runs, so
`#[tool]` reads the same two things while the code is still source:

```rust
#[tool]
/// Look one term up in the index and return what it says.
///
/// Give a single term, not a sentence.
fn search(term: String) -> anyhow::Result<String> {
    Ok(format!("nothing on {term}"))
}
```

The function stays callable as itself, and the tool is a type of the same name in PascalCase —
`vec![Box::new(Search)]` — while the name on the wire is still `search`. The doc comment **is
prompt text**: it reaches the model as `search, whose description is <desc>Look one term up in the
index and return what it says. ...</desc>`, normalised by the same `inspect.cleandoc` Python
applies to a docstring, so an indented second paragraph reads the same in both languages. The
parameters become the schema `{"term": {"type": "string"}}` — an `Option<T>` is one the model may
leave out.

**A wrong argument is answered, not raised.** Sending `{"term": 7}` gets back ``Refused: `term` is
not the type this tool takes (invalid type: integer `7`, expected a string).`` — a string the loop
can read and retry from, where an error would end the turn. dspy's tools answer the same way.

### A tool that needs state

Python captures a draft in a closure and hands six closures to `dspy.ReAct`. A Rust `fn` captures
nothing, so the state is the receiver instead: mark a method and you get a tool beside it, named
for the method.

```rust
struct Composition {
    draft: Mutex<String>,
}

impl Composition {
    /// Append one paragraph to the draft.
    ///
    /// Write the paragraph itself, not a description of it.
    #[tool]
    fn append(&self, text: String, heading: Option<String>) -> anyhow::Result<String> {
        let mut draft = self.draft.lock().expect("the draft outlives the roster");
        if let Some(heading) = heading {
            draft.push_str(&format!("## {heading}\n"));
        }
        draft.push_str(&text);
        Ok(format!("Added {} characters.", text.len()))
    }
}
```

```rust
let composition = Arc::new(Composition { draft: Mutex::new(String::new()) });
let agent = ReActV2!("brief -> written: bool", vec![composition.append_tool()]);
```

`append` stays an ordinary method you can call and test directly; `append_tool()` is the same thing
as something `ReAct` can hold. Interior mutability is yours, exactly as it is in Python: the tool
outlives any one call, so the draft is behind a `Mutex`.

**Put the attribute on the impl block too and you get the whole roster at once** — every marked
method, in declaration order, as `tools()`. That is all it adds, and with six tools it is the
difference between one call and six:

```rust
struct Section {
    draft: Mutex<String>,
}

#[tool]
impl Section {
    pub fn new() -> Self { Section { draft: Mutex::new(String::new()) } }

    // Unmarked: an ordinary method, not something the model can call.
    fn len(&self) -> usize {
        self.draft.lock().expect("the draft outlives the roster").len()
    }

    /// Append one paragraph to the section.
    #[tool]
    fn write(&self, text: String) -> anyhow::Result<String> {
        self.draft.lock().expect("the draft outlives the roster").push_str(&text);
        Ok(format!("Added {} characters.", text.len()))
    }

    /// Read the section back as written so far.
    #[tool]
    fn read(&self) -> anyhow::Result<String> {
        Ok(self.draft.lock().expect("the draft outlives the roster").clone())
    }
}
```

```rust
let section = Arc::new(Section::new());
let agent = ReActV2!("brief -> written: bool", section.tools());
```

### A tool that has to wait

A tool that reaches a network, a database or a subprocess is a future. Write it `async` and the
attribute does the rest — dspy's `Tool.acall`, which awaits a tool whose callable is a coroutine:

```rust
#[tool]
/// Fetch what one URL says.
async fn fetch(url: String) -> anyhow::Result<String> {
    // Your HTTP client goes here. What matters is that the body awaits.
    tokio::task::yield_now().await;
    Ok(format!("whatever {url} answered"))
}
```

Every agent awaits every tool, so a roster can mix the two freely: a synchronous tool answers on
the awaited path unchanged, which is upstream's own "allow calling a sync tool in the async path".
An `async` tool answers only there — calling it synchronously says so rather than blocking a
runtime thread.

### Without the attribute at all

`#[tool]` exists for one reason: a Rust doc comment is erased before the program runs, so nothing
can read it later. But that is only true of a comment on a *function*. On a **type**, the
`JsonSchema` derive records it — and records the fields, their types, and their own doc comments
with it. So a tool can be built out of derives this crate did not write:

```rust
/// Append one instructional block to the draft and return its id.
#[derive(Deserialize, JsonSchema)]
struct AddBlock {
    /// One of: explanation, worked_example.
    block_type: String,
    text: String,
}

let add_block = typed_tool(move |args: AddBlock| held.add_block(args.block_type, args.text));
```

`typed_tool` reads all four off the type: the name from `AddBlock`, the description from its doc
comment, the arguments from its fields, and each argument's prose from that field's doc comment.
The closure captures, so a roster of these shares one state the way Python's closures do.

It costs a struct per tool and buys **prose per argument** — dspy's `Tool.arg_desc`, which
`#[tool]` on a function cannot give you, because a Rust parameter takes no doc comment. Reach for
it when an argument needs explaining; reach for `#[tool]` when the tool's own description is
enough.

For a tool whose schema is only known at run time — one built from an MCP server, say — `FnTool::new`
takes the name, description and schema as values and skips the reflection entirely.

**Name a task and the answer is that task's own struct** — the same line whichever module wrote
it, which is what dspy gets from `Prediction.__getattr__` and this gets from the type:

```rust
let out = call!(Predict!(QA), question = "capital of France?").await?;
let out = call!(ChainOfThought!(QA), question = "capital of France?").await?;
let out = call!(ReActV2!(QA, tools), question = "capital of France?").await?;
let answer = out.answer;                                  // checked when this compiles
```

Name a *string* signature instead and there is no type to answer with, so it stays a `Prediction`
and `out.get("answer")` reads it — as does `out.typed::<QAOutputs>()?`, which checks the whole
answer at once and ignores fields the task never declared.

A module of your own says which task it answers with beside the derive, and then reads the same
way — there is nothing to add at the call site:

```rust
#[derive(Module)]
#[task(QA)]
struct AskTwice {
    first: Predict,
    second: Predict,
}

impl Forward for AskTwice {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let draft = self.first.forward(inputs).await?;
        self.second.forward(draft.example).await
    }
}
```

```rust
let twice = AskTwice {
    first: Predict!("question -> answer"),
    second: Predict!("question -> answer"),
};
let out = call!(twice, question = "capital of France?").await?;
let answer = out.answer;
```

Leave `#[task(..)]` off and it answers with a `Prediction`, which is what a program whose outputs
are not one task's has to answer with. An already-built module takes the same step by hand —
`program.task::<QA>()` — which is exactly what the macros above call for you.

The module underneath is still there: `agent.max_iters` reads through, and `into_module()` hands it
back when you want the `Prediction` an agent's `trajectory` and `history` live in. A builder
consumes the module, so it goes through `map`:

```rust
let agent = ReActV2!(QA, tools).map(|agent| agent.max_iters(8));
```

**An agent can finish with nothing**, which is the case `Predict` never reaches: it ran out of
turns, or answered in prose instead of calling `submit`. That is a successful loop with no outputs
in it, so the call fails and names what happened rather than the missing field:

    the loop ended without producing the outputs (max_iters)

dspy raises `AttributeError` at the moment `out.answer` is touched, which says the same thing later
and with less in it.

Every signature-taking macro accepts **either spelling** — a string, or a task declared with
`#[derive(Signature)]`:

```rust
let quick   = Predict!("question -> answer");
let declared = Predict!(Investigate);          // carries its doc comment as instructions
let agent    = ReAct!(Investigate, tools, max_iters = 4);
let reader   = RLM!(Investigate, max_iters = 6);
```

Every module spells its cap `max_iters`, as DSPy 3.3.0 does. It did not until then — `RLM` used
`max_iterations` through 3.3.0b1, and this crate carried the split rather than inventing a uniform
name neither side had. The pin moved and so did this.

### Code-writing modules run real Python

`ProgramOfThought`, `CodeAct` and `RLM` default to a Deno/Pyodide sandbox running DSPy's own
`runner.js`, exactly as `dspy.ProgramOfThought(...)` defaults to `PythonInterpreter()`. **`deno`
must be on the path**, which is what DSPy asks of its users too:

```bash
curl -fsSL https://deno.land/install.sh | sh    # or: brew install deno
```

Supply your own environment with `interpreter`, which is also how a test scripts one without
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
let qa = ChainOfThought!("question -> answer");
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

let qa = ChainOfThought!("question -> answer");
let best = BestOfN!(qa, n = 3, reward = one_word, threshold = 1.0);
let out = call!(best, question = "capital of Belgium?").await?;
```

The reward is a named function in both, which is what dspy's own example does. A Rust closure
works too — `|inputs: &Example, out: &Prediction| …` — but writing one inline inside the
constructor buries the three arguments around it.

**`BestOfN!` names the arguments** because dspy passes all four by keyword and Rust has no named
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
| Constructor | `dspy.Predict(x)` | `Predict!(x)` — a string or a task |
| Call | `m(field=…)` | `call!(m, field = …)` |
| Reading a result | `out.answer` | `out.answer` from any module named on a task; `out.get("answer")` from a string signature |
| Asking the model directly | `lm(dspy.User("…"))` | `lm.call(items![User(["…"])])` |
| A turn | `dspy.User(…)` | `User([…])`, or `LmMessage::user([…])` |
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
supplies a list literal. `Predict!` additionally checks a signature string while the crate
compiles, which is something dspy cannot do at all.
