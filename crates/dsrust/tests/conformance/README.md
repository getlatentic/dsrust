# Upstream conformance fixtures

Goldens: what a pinned Python DSPy actually produces. Every fixture here is written by
*running* dspy, never by transcribing an assertion out of its test files — a hand-copied
expectation only ever tests the copying.

Regenerate with the pinned interpreter, which `scripts/DSPY_VERSION` names:

```sh
.dspy-venv/bin/python scripts/generate_fixtures.py     # the *.json beside this file
.dspy-venv/bin/python scripts/generate_rng_fixture.py  # rng/cpython_random.json
```

## `*.json` — rendered prompts

One signature, its demos, and its input values, with the system message and turns dspy's
`ChatAdapter` renders for them. `tests/conformance.rs` globs this directory, so a new fixture is
exercised the moment it lands. Add cases to `CASES` in `scripts/generate_fixtures.py`.

That glob is why every other fixture lives in a subdirectory: a `.json` dropped beside this file is
read as a rendered prompt, and one that is not fails on a missing `inputs` array.

## `observe/` — which callback handlers a program fires, and under what

`observe/callbacks.json` is the handler sequence four dspy programs produced under a recording
`BaseCallback`, with the nesting depth of the call each handler belongs to. The payloads do not
cross — upstream hands a handler Python objects — so what is held here is *when* each point fires
and what encloses it, which is the half a trait of defaulted methods gets wrong by counting.
`tests/callback.rs` reads it; `scripts/generate_callback_fixture.py` writes it. The
`chain_of_thought_n3` case is the one sequence upstream asserts by hand, in
`tests/callback/test_callback.py`.

## `lm_api/` — dspy 3.3's normalized LM types

`lm_api/dspy_3_3.json` is one pydantic dump per class, every field included rather than only the
set ones, so a field this crate fails to model shows up as a parse failure. `tests/
lm_api_conformance.rs` reads it. Regenerate with the 3.3 interpreter, which is deliberately a
second venv — 3.2.1 stays the pin for rendered bytes:

```sh
uv venv .dspy-venv-3.3 --python 3.12
uv pip install --python .dspy-venv-3.3/bin/python "dspy==3.3.0b1"
.dspy-venv-3.3/bin/python scripts/generate_lm_api_fixture.py
```

Outside the glob above, since it has a different shape and would fail the prompt harness's
`inputs array` read.

## `rng/` — the generator behind an optimizer's choices

dspy seeds `random.Random(0)`, so which examples an optimizer keeps is decided by CPython's
Mersenne Twister. `rng/cpython_random.json` records what that generator draws, and
`src/optimize/rng.rs` is held to it.

That subdirectory is deliberately not part of the glob above — it has a different shape, and the
Rust that reads it is private to `src/optimize`, so the check lives in that module's own tests
rather than in `tests/conformance.rs`.
