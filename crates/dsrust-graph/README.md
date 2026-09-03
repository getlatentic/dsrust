# Building a dsrust program from a graph wired at run time

A worked reference for the case a visual builder lands in: the program's shape comes from a
**document** — nodes and edges a user drew — not from a Rust struct.

```sh
cargo run --example run -- --document                    # the document shape
cargo run --example run -- --subject "winter mornings"   # run it
cargo run --example run -- --optimize                    # compile it
```

Copy `src/graph.rs`. It is two methods.

## What this needs from dsrust

**Attribution is the engine's, and this crate depends on it.** `Predict::forward` records each
call into a task-local buffer and names it from the walk, so an optimizer can tell which node
earned which demo. `Graph` used to hand-write `forward_traced` to supply that; against an engine
that does not record ambiently, it no longer does — and an optimizer then reflects on nothing,
proposes nothing, and reports success having changed nothing. Silently.

That landed in `dsrs` as `fb2c195` on **`pin/dspy-3.3.0`**, and the dependency is by path — so the
branch checked out over there is what this builds against.

**Only that branch is a build target.** `main` is 466 commits behind it, last moved 2026-07-24, and
pins `scripts/DSPY_VERSION` to the beta `3.3.0b1` where this branch is on `3.3.0` final; `dev` is
208 behind. Against those, this is not "the same engine minus attribution" — it is a port a month
and hundreds of commits older, and attribution is one absence among many. Every one of those
branches is *0 behind*, so advancing them is a fast-forward with nothing to reconcile; until that
happens, build against `pin/dspy-3.3.0`.

## When this models tools

`ReAct` and `CustomModule` are refused rather than built, because a `Step` holds a `Predict` and
this translation models no tools — a node built as a plain Predict answers without ever calling
one, which looks identical to a working agent in every output. Refusing is the placeholder; the
engine side is already there when someone comes to lift it.

dsrust has `ReAct` (classic text protocol) and `ReActV2` (native tool calling), and `#[tool]` reads
a Rust fn's doc comment and typed parameters into the roster the model sees. `#[tool(default = …)]`
makes an argument optional the way dspy leaves a defaulted parameter out of `required`;
`#[tool(desc = "…")]` states a description outright, which is needed whenever the text carries
indentation, since rustdoc compiles an indented doc line as a doctest.

The names to expect through the walk: dspy calls ReAct's two predictors `react` and
`extract.predict`, and dsrust matches. So a node `n1` holding one yields `n1.react` and
`n1.extract.predict`, and a state file keyed by those loads either way.

One difference that matters only if generated code inspects module *structure* rather than the
prompt: dsrust flattens the extraction step to a `Predict` over a signature that already carries
`reasoning`, where dspy holds a `ChainOfThought` there. The prompt is byte-identical and held to a
golden; only the name says `extract.predict`.

`cargo test` is the check: `a_compile_actually_changes_the_graph` and
`gepa_writes_back::a_gepa_compile_rewrites_the_graphs_instructions` both fail without attribution,
which is the whole reason they assert on what changed rather than on `compile` returning `Ok`.

## The finding, first

`#[derive(Module)]` writes a program from a struct's **fields** — and, from that same field list,
the walk an optimizer works through. That list is fixed when the crate compiles.

A graph's nodes are a `Vec` whose length is known when the document loads. **So the derive cannot
help, and `Module` is written by hand.** That is the one case where hand-writing is right rather
than a mistake.

It is also where hand-writing costs most, because `Module::named_predictors` **defaults to
answering with nothing**:

| | `on_module_start` | `named_predictors` | `compile` returns |
|---|---|---|---|
| `#[derive(Module)]` | fires | every step | `Ok`, having rewritten them |
| hand-written, both methods | fires | every step | `Ok`, having rewritten them |
| hand-written, `forward` only | **silent** | **empty** | **`Ok`, having rewritten nothing** |

That last row is the trap. The optimizer walks an empty list, rewrites nothing, and reports
success. The progress bar fills. The saved program is byte-identical to the one you started with.
Nothing anywhere says so.

`tests/graph.rs` holds it, and the assertion is on the demos the walk earned rather than on
`compile` returning `Ok` — because `Ok` passes either way. Delete `named_predictors` and three
tests fail; keep it and five pass.

## The two methods

**`forward` — the wiring is the forward.** Each node's inputs are read from the program's own
inputs or from an earlier node's answer, in declaration order. No user-written code runs, which is
why a declarative canvas needs no runtime and no Python.

**`named_predictors` — every node, named after itself.** Naming each after its node id is what lets
a caller show a rewritten instruction against the box on the canvas it belongs to.

## Two ways a node declares what it asks for

A builder that writes signatures as strings uses the first. **A canvas with a node per field uses
the second** — and has to, because once a field carries a type the string spelling cannot express,
there is no string to parse.

```json
{ "id": "plan",  "signature": "subject -> angle" }

{ "id": "write",
  "inputs":  [{ "name": "angle",     "type": "str", "description": "The angle to write on." }],
  "outputs": [{ "name": "citations", "type": "list[Citation]" }] }
```

`str`, `bool`, `int`, `float` and `reasoning` name themselves. **Anything else is carried
verbatim as the field's annotation** — `list[str]`, `dict[str, Any]`, a custom type's own name —
which is exactly how a custom type reaches dspy: the annotation is printed and the value travels as
JSON. So a canvas can offer any type it likes without the builder needing to know them.

The example document uses one of each, so every test exercises both paths.

## Check the walk before you trust it

```rust
graph.walk_covers_every_node()?;      // before compiling, not after
```

The failure this guards is silent: a program whose `named_predictors` misses a node is one the
optimizer walks past, and `compile` still returns `Ok`. It reports which:

    the optimizer would walk 1 of 2 nodes, missing ["plan"] — a compile would report
    success having rewritten nothing

There is a second reason to call it, for anyone showing an instruction diff. **An optimization that
walked every node and still changed nothing is a real outcome** — worth showing rather than
celebrating — and it is indistinguishable from this bug unless the walk was checked first.

## Calibrate's own document

`src/calibrate.rs` reads Calibrate's real graph — the fixtures in `tests/fixtures/` are copied from
`calibrate-codegen/tests/fixtures/`. Three things about that shape decide the translation, and each
has a test:

**A field can be renamed across an edge.** `predict.field_out` carries `fieldName: "answer"` into
`cot.field_in` as `fieldName: "context"`. So a wire is **named by the receiving end and fetched by
the sending end** — getting that backwards feeds a module a field it never produced, silently, as a
null.

**A module's signature comes from its edges**, not from a declaration of its own: inputs are the
incoming data edges' `to.fieldName`, outputs the outgoing edges' `from.fieldName`. In the fixture
that makes `predict` a `question -> answer` and `cot` a `context -> summary`.

**The answer is the OutputField layer.** Each `OutputField` names one field of the program's answer
and is fed by one module field — which is why `answers` is a `Vec` rather than one node id.

**Naming precedence is asymmetric, and it is the easiest thing here to get wrong.**

| | wins | falls back to |
|---|---|---|
| a module's **input** field | the edge's `to.fieldName` | the `InputField` node's `config.name` |
| a module's **output** field | the `OutputField` node's `config.name` | the edge's `from.fieldName` |

Opposite directions, deliberately: Calibrate's own resolver takes the node for an output *so a
rename propagates immediately* rather than waiting for the edge to catch up. Take the edge's word
for it and a renamed output builds a program still answering under the old name.

Both directions have a test where the two **disagree**, because a test where they agree passes
under either rule and proves nothing.

Both spellings stay valid — `fieldName` is optional by design, absent on older edges and on
boundary ports, which carry one field per port by construction. `seed_field_node_graph.json` names
its fields at neither end of any edge, so a name falls back to a boundary node at *either* end
before the loader gives up.

**A node kind this builder does not know stops the load.** Filtering to the module kinds and
moving on is what a builder does naturally, and it is wrong: a kind added to Calibrate and not here
would be silently dropped, and what got built would be a program the document does not describe —
fewer steps, running less and optimizing less, with nothing saying so. Measured before the check
existed: renaming one module's kind built a **one-node program whose answer pointed at the node
that had been dropped**. A new kind is now a change to `MODULES` or `BOUNDARY` and a decision about
what it means, which is the point.

`edgeKind: "control"` orders execution and carries no data, so it is read for nothing — order comes
from the node list, which Calibrate emits topologically.

Instructions are the module's `SignatureSpec.docstring`, falling back to a line naming the node
when the document carries none. Both fixtures carry none, so both run on the fallback.

## The built-in document

Deliberately small — enough to be a real graph, and nothing belonging to any particular editor:

```json
{
  "nodes": [
    { "id": "plan",  "signature": "subject -> angle",
      "instructions": "Pick one angle on the subject.",
      "inputs": [{ "name": "subject", "from": "input", "field": "subject" }] },
    { "id": "write", "signature": "angle -> haiku",
      "instructions": "Write a haiku on that angle.",
      "inputs": [{ "name": "angle", "from": "node", "node": "plan", "field": "angle" }] }
  ],
  "answer": "write"
}
```

A richer document maps onto this by ignoring the rest. Two things worth keeping whatever shape
yours takes:

- **`instructions` on the node.** They are what an optimizer rewrites and what you write back. A
  document without them has nothing to compile.
- **Signatures parsed at load.** A bad one fails when the document opens rather than at the first
  run — the closest a runtime-shaped program gets to the build-time check
  `Predict!("subject -> angle")` would have given it.

## What this does not do

**Execution order is declaration order.** This is an interpreter, not a scheduler: a document that
wires a node from a later one reads as a missing value rather than deadlocking. A builder should
reject that when it saves, and a topological sort here would be the alternative.

**Nothing is parallel.** Two nodes with no edge between them still run in sequence. `dsrust::Parallel`
is the seam if a canvas has genuine branches.

**No imperative bodies.** A node is a signature. A node whose forward is hand-written code is the
one thing a declarative document cannot express — and the only place a language runtime would enter.

## Live

Against `deepseek-v4-flash`:

```
subject   winter mornings
haiku     Frosted pane at dawn,
          each breath a cloud in still air—
          day holds its promise.
```

Both nodes ran; the second was fed the first's answer.
