#!/usr/bin/env python3
"""Whether litellm is still the only oracle for the providers grounded on it.

`lm_api/litellm_chat.json` records the request body litellm puts on the wire for Anthropic and
ollama, and the Rust builders assert byte equality against it. That is correct for one reason and
one only: **dspy 3.3 ships native wire code for OpenAI alone.** `clients/openai_format.py` maps the
typed `LMRequest` to OpenAI-shaped JSON without calling a provider, and there is no sibling doing
the same for Anthropic or ollama — so for those two, what litellm sends *is* what dspy sends, and
there is nothing else to be faithful to.

That reason has an expiry date. litellm is already a lazy import behind `require(..., extra=
"litellm")` in 3.3, dspy's own migration note puts the legacy types gone by 4.0, and the direction
of travel is typed types plus native per-provider wire. The day dspy ships `to_anthropic_request`
or an ollama equivalent, this golden is grounded on the wrong upstream — and nothing would say so,
because the fixture would still regenerate, the tests would still pass, and the bytes would still
match litellm. It would simply have stopped matching *dspy*.

So this asserts the condition rather than the consequence: no native non-OpenAI wire in the pinned
dspy. When that fails, the fix is not to silence it — it is to regenerate the affected cases from
dspy's own converter and move the golden's `_source` to say so.

    .venv/bin/python scripts/check_litellm_grounding.py
"""

from __future__ import annotations

import pathlib
import re
import sys

import dspy

#: The providers whose bytes this repository takes from litellm, and the spellings a native dspy
#: converter for them would plausibly use. Deliberately broad: a false alarm costs one reading of
#: one new file, and a miss costs a silently wrong oracle.
GROUNDED_ON_LITELLM = {
    "anthropic": ("anthropic",),
    "ollama": ("ollama",),
}

#: dspy's own typed-wire module, named so the check fails loudly if it is ever renamed rather than
#: quietly finding no siblings and passing.
TYPED_WIRE = "openai_format.py"

CONVERTER = re.compile(r"^\s*def\s+(to_(\w+)_request|(\w+)_to_lm_response)\b", re.M)


def clients_dir() -> pathlib.Path:
    return pathlib.Path(dspy.__file__).parent / "clients"


def native_wire_modules(root: pathlib.Path) -> list[str]:
    """`*_format.py` siblings of the OpenAI one — the shape a native wire would arrive in."""
    return sorted(
        path.name
        for path in root.glob("*_format.py")
        if path.name != TYPED_WIRE
    )


def converters_for(root: pathlib.Path) -> dict[str, list[str]]:
    """Any `to_<provider>_request` / `<provider>_to_lm_response` naming a grounded provider."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in CONVERTER.finditer(text):
            name = match.group(1)
            for provider, spellings in GROUNDED_ON_LITELLM.items():
                if any(spelling in name.lower() for spelling in spellings):
                    found.setdefault(provider, []).append(f"{path.name}::{name}")
    return found


def main() -> int:
    root = clients_dir()
    if not (root / TYPED_WIRE).exists():
        print(
            f"litellm-grounding check FAILED:\n"
            f"  {TYPED_WIRE} is gone from {root}. The typed wire this check is defined against has "
            f"moved, so its answer means nothing until someone re-reads it.",
            file=sys.stderr,
        )
        return 1

    siblings = native_wire_modules(root)
    converters = converters_for(root)
    if not siblings and not converters:
        print(
            f"  litellm grounding: OK (dspy ships {TYPED_WIRE} and no native wire for "
            f"{', '.join(sorted(GROUNDED_ON_LITELLM))})"
        )
        return 0

    print("litellm-grounding check FAILED:", file=sys.stderr)
    print(
        "  dspy now ships wire code for a provider this repository takes from litellm, so "
        "`lm_api/litellm_chat.json` is grounded on the wrong upstream.",
        file=sys.stderr,
    )
    for name in siblings:
        print(f"      new wire module: clients/{name}", file=sys.stderr)
    for provider, names in sorted(converters.items()):
        for name in names:
            print(f"      {provider}: {name}", file=sys.stderr)
    print(
        "  Regenerate those cases from dspy's converter and say so in the golden's `_source`.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
