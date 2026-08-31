#!/usr/bin/env python3
"""A macro names its dependencies through this crate, never through the caller's.

`quote! { ::anyhow::Result<_> }` resolves in the *expansion site's* crate root. That is fine while
every test lives in a workspace that happens to depend on `anyhow`, and broken the moment someone
writes a crate that depends on nothing but `dsrust` — which is exactly what README.md promises they
can do.

It has happened three times, each found by an outside caller rather than by a gate:

  * `#[derive(Signature)]` named `::serde`, reported from a real project.
  * `#[tool]` named `::serde_json` and `::anyhow`, found while writing a tool in a fresh crate.
  * `#[derive(Module)]` named `::anyhow`, found while porting a DSPy tutorial — after the other two
    were fixed, in the one macro the external-consumer gate never exercised.

`check_external_consumer.sh` catches this for the macros it uses, and only those. This reads what
every macro emits, so a macro nobody thought to exercise is covered by the same rule.

    ./scripts/check_macros_route_their_paths.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Crates a macro may need in its expansion. Each is re-exported from `dsrust::__macro_support`,
#: which is what a routed path names.
ROUTED = "anyhow|serde|serde_json|schemars|tokio|futures|reqwest|tracing"

#: `::crate::` that is not already reached through this crate. `$crate::` is `macro_rules!`'s own
#: routing and is correct; `__macro_support::` is the proc-macro spelling of the same thing.
UNROUTED = re.compile(rf"(?<!__macro_support)(?<!\$crate)::({ROUTED})::")


def quoted(text: str) -> str:
    """The bodies of every `quote!` block — what a proc macro puts in the caller's crate."""
    out = []
    for opened in re.finditer(r"quote!\s*\{", text):
        depth, at = 0, opened.end() - 1
        while at < len(text):
            if text[at] == "{":
                depth += 1
            elif text[at] == "}":
                depth -= 1
                if depth == 0:
                    break
            at += 1
        out.append(text[opened.end() : at])
    return "\n".join(out)


def exported_arms(text: str) -> str:
    """The arms of every `#[macro_export] macro_rules!`, which expand the same way."""
    return "\n".join(
        found.group(1)
        for found in re.finditer(
            r"#\[macro_export\][^!]*?macro_rules!\s+\w+\s*\{(.*?)\n\}", text, re.S
        )
    )


def main() -> int:
    print("==> Every macro routes its paths through this crate")
    unrouted: list[str] = []
    checked = 0
    for source in sorted((ROOT / "crates/dsrust-derive/src").rglob("*.rs")):
        body = quoted(source.read_text())
        checked += bool(body)
        for line in body.splitlines():
            if UNROUTED.search(line):
                unrouted.append(f"{source.relative_to(ROOT)}: {line.strip()[:84]}")
    for source in sorted((ROOT / "crates/dsrust/src").rglob("*.rs")):
        body = exported_arms(source.read_text())
        checked += bool(body)
        for line in body.splitlines():
            if UNROUTED.search(line):
                unrouted.append(f"{source.relative_to(ROOT)}: {line.strip()[:84]}")

    print(f"    macro-emitting files read: {checked}")
    if unrouted:
        print(f"\nMacro-routing gate FAILED: {len(unrouted)} path(s) the caller must resolve:")
        for hit in unrouted:
            print(f"    {hit}")
        print(
            "\n  Name it through `::dsrust::__macro_support::` (proc macro) or `$crate::`"
            "\n  (macro_rules!), so a crate depending only on dsrust can expand it."
        )
        return 1
    print("Macro-routing gate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
