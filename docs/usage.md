# Declaring a task, and asking it

Every Rust shape here is compiled and run by [`tests/every_spelling.rs`](../tests/every_spelling.rs).
If one of them stops being true, that test fails rather than this page quietly going stale.

A task is declared one of two ways — its field names in a string, or a struct — and asked by one
of two modules. Both declarations produce the same type and are asked the same way, which is what
dspy gives by having one `Predict` class and one `Signature` base.

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

Every named field is a step, so `named_predictors` — the seam an optimizer works through — comes
from the field list, and each child's predictors are renamed after the field holding them, so a
demo says which step earned it. A field that is not a step carries `#[not_a_step]`. The derive
also makes the module callable through `call!`.

`Forward` exists so an author is not writing `Pin<Box<dyn Future>>` by hand. `Module` keeps that
shape because it must be object-safe for a composed program to hold `Box<dyn Module>`; the derive
does the boxing in between.

## Every module, and what it takes

dspy's modules split into two families, and the split decides how you build one.

**A module that takes a signature** is declared like `Predict` — a field-name string or a task
type — and each has a macro of the same name.

**A wrapper takes another module.** There is no signature to hand it; the signature lives in
whatever it wraps, so it is built with `::new` and has no macro.

| module | takes | dspy | dsrs |
|---|---|---|---|
| `Predict` | a signature | `dspy.Predict("q -> a")` | `predict!("q -> a")` |
| `ChainOfThought` | a signature | `dspy.ChainOfThought("q -> a")` | `chain_of_thought!("q -> a")` |
| `ReAct` | a signature + tools | `dspy.ReAct("q -> a", tools=[…])` | `ReAct::new(signature!("q -> a"), tools)` |
| `MultiChainComparison` | a signature | `dspy.MultiChainComparison("q -> a")` | `MultiChainComparison::new(…)` |
| `BestOfN` | **a module** | `dspy.BestOfN(module=qa, N=3, reward_fn=f, threshold=1.0)` | `BestOfN::new(qa, 3, f, 1.0)` |
| `Parallel` | branches per call | `dspy.Parallel(num_threads=8)` | `Parallel::new(8)` |

`Refine`, `ProgramOfThought`, `CodeAct` and `RLM` are not built yet. The first is a wrapper like
`BestOfN`; the other three take a signature and will get macros.

### Asking one, side by side

```python
# dspy
qa = dspy.ChainOfThought("question -> answer")
out = qa(question="capital of France?")
out.answer
```

```rust
// dsrs
let qa = chain_of_thought!("question -> answer");
let out = call!(qa, question = "capital of France?").await?;
out.get("answer").unwrap()
```

### Wrapping one

`BestOfN` runs a module up to `n` times, each at a fresh rollout and `temperature = 1.0`, and
keeps the highest-scoring attempt — stopping early at the first to reach `threshold`.

```python
# dspy
qa = dspy.ChainOfThought("question -> answer")
best = dspy.BestOfN(module=qa, N=3, reward_fn=one_word, threshold=1.0)
out = best(question="capital of Belgium?")
```

```rust
// dsrs
let best = BestOfN::new(
    predict!("question -> answer"),
    3,
    |_inputs: &Example, out: &Prediction| match out.get("answer").and_then(|a| a.as_str()) {
        Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
        _ => 0.0,
    },
    1.0,
);
let out = call!(best, question = "capital of Belgium?").await?;
```

All four arguments are upstream's, in its order. Two details are upstream's too and easy to read
as bugs: `threshold` is **required**, because `BestOfN.forward` compares against it with no
guard; and `with_fail_count(0)` means *n*, not *none allowed*, because dspy reads
`fail_count or N` and Python treats zero as unset.

A wrapper is still a module — it nests, `call!` reaches it, and an optimizer's walk goes straight
through to the predictors inside it.

## Against dspy

| | dspy | dsrs |
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
