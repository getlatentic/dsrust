"""Record the constant tables the crate carries that no rendered prompt would reveal.

Prompt prose is held by goldens because a fixture renders it. A table that only ever drives a
*decision* — which names an interpreter refuses, which headers a request id hides in — is invisible
to every fixture the crate has, so it sits transcribed. `TIPS` was one of those, and it was right;
the point is that nothing would have said so.

Two are recorded here, each from its own oracle rather than from a reading:

  - the names `_inject_variables` refuses. Upstream does not hold a list at all — it asks
    `keyword.iskeyword(key) or key == "json"`, so the oracle is CPython's own `keyword.kwlist` plus
    upstream's one addition. Being a list rather than a predicate is itself a divergence: a keyword
    added by a later CPython would reach dspy and not the crate, which is worth knowing and is why
    the interpreter version is recorded beside it.
  - each adapter's own name. dspy dispatches a callback by the instance's *type* and hands the
    handler the instance, so what a watcher sees is `type(instance).__name__`. A Rust callback
    cannot hand over a `dyn Adapter` for the caller to downcast, so `Adapter::name` stands in — and
    it is only a stand-in if it says what upstream's class is called. Three of five did not
    (`JsonAdapter` for `JSONAdapter`, and so on), found because nothing asserted any of them.
  - the headers `_exception_request_id` tries, in its order. They are literals inside the function
    body, so they are read out of dspy's own AST rather than copied.

`adapter/native_tools.rs`'s `TOOL_ANNOTATIONS` is deliberately not here: dspy compares annotations
structurally (`origin is list and args[0] == Tool`), so those two strings are this crate's own
spelling of a type test and there is no upstream string to pin them against.

    .venv/bin/python scripts/generate_constants_fixture.py
"""

from __future__ import annotations

import ast
import inspect
import json
import keyword
import pathlib
import platform
import sys

from dspy.clients import lm as lm_module

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "constants"
)
PINNED = require("dspy")


def string_constants(function) -> list[str]:
    """Every string literal in a function's body, in source order."""
    tree = ast.parse(inspect.getsource(function).lstrip())
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def adapter_names() -> dict[str, str]:
    """Each adapter class's own name, keyed by the wire this crate calls it.

    Read off the classes rather than written down: the names are what `type(instance).__name__`
    gives a callback handler, and a transcribed list would agree with whatever it was transcribed
    from.
    """
    import dspy
    from dspy.adapters.baml_adapter import BAMLAdapter

    return {
        "chat": dspy.ChatAdapter.__name__,
        "json": dspy.JSONAdapter.__name__,
        "xml": dspy.XMLAdapter.__name__,
        "baml": BAMLAdapter.__name__,
        "two_step": dspy.TwoStepAdapter.__name__,
    }


def main() -> None:
    # `_inject_variables` refuses a name when `keyword.iskeyword(key) or key == "json"`.
    refused = sorted(keyword.kwlist) + ["json"]

    headers = string_constants(lm_module._exception_request_id)
    if len(headers) != 4:
        raise SystemExit(f"expected four request-id headers, read {headers!r}")

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_constants_fixture.py",
        "dspy_version": PINNED,
        "python_version": platform.python_version(),
        "note": (
            "Tables that drive a decision rather than a prompt, so no rendered golden sees them. "
            "The refused names come from CPython's keyword.kwlist plus dspy's own 'json', because "
            "upstream asks keyword.iskeyword rather than holding a list — a keyword added by a "
            "later CPython would reach dspy and not a static list."
        ),
        "adapter_names": adapter_names(),
        "refused_variable_names": refused,
        "request_id_headers": headers,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tables.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    print(
        f"  {len(refused)} refused names (python {platform.python_version()}), "
        f"{len(headers)} request-id headers",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
