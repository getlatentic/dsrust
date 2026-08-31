#!/usr/bin/env python3
"""Every type a public signature names is a type a caller can name.

A `pub fn` whose parameter or return type lives in a private module compiles, documents, and is
unusable: the caller can see the method and cannot write the argument. Nothing else here catches
that, because both halves are individually fine — the method is public, and the type is `pub`
inside a module that is not.

Two have been found by hand, both by someone writing a real program against the crate:

  * `MIPROv2::compile_traced` answers with `Vec<Trial>` and `Trial` was unreachable — a value a
    caller could hold and not name.
  * `MIPROv2::auto` takes an `Auto` and `Auto` was unreachable — so the method could not be called
    at all. That one survived a full gate run and was found by porting a DSPy tutorial.

Both are the same shape, and one was fixed without anyone asking whether there were others. This
asks.

Needs a nightly toolchain for rustdoc's JSON output, as `check_surface_against_rustdoc.py` does,
so it is run rather than gated:

    ./scripts/check_public_types_are_nameable.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Types whose own crate is not ours: a caller names them through their own dependency, or through
#: a re-export this crate already makes, and either way their reachability is not ours to assert.
FOREIGN = "foreign"


def rustdoc(package: str) -> dict:
    built = subprocess.run(
        ["cargo", "+nightly", "doc", "--no-deps", "-p", package],
        cwd=ROOT,
        env={**os.environ, "RUSTDOCFLAGS": "-Z unstable-options --output-format json"},
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        raise SystemExit(f"rustdoc JSON needs a nightly toolchain:\n{built.stderr[-600:]}")
    return json.loads((ROOT / "target" / "doc" / f"{package}.json").read_text())


def type_ids(node: object, into: set[int]) -> None:
    """Every type id a signature mentions, however deeply nested in generics."""
    if isinstance(node, dict):
        path = node.get("resolved_path")
        if isinstance(path, dict) and isinstance(path.get("id"), int):
            into.add(path["id"])
        for value in node.values():
            type_ids(value, into)
    elif isinstance(node, list):
        for value in node:
            type_ids(value, into)


def signatures(index: dict[int, dict]) -> list[tuple[str, dict]]:
    """Public functions and methods, named by where a caller meets them."""
    found = []
    for item in index.values():
        if item.get("visibility") != "public" or not item.get("name"):
            continue
        inner = item.get("inner", {})
        if "function" in inner:
            found.append((item["name"], inner["function"].get("sig", {})))
    return found


def main() -> int:
    print("==> Every type a public signature names is nameable")
    doc = rustdoc("dsrust")
    index = {int(k): v for k, v in doc["index"].items()}
    paths = {int(k): v for k, v in doc["paths"].items()}
    local = index[doc["root"]]["crate_id"]

    # Nameable = reachable by walking public items from the crate root, *following re-exports*.
    # rustdoc lists an item by where it is defined, so a `pub use` is the whole difference between
    # a type in a private module that a caller can name and one it cannot — reading definition
    # paths alone reports every re-exported type as hidden, which is a checker that cries wolf.
    reachable: set[int] = set()
    frontier = [doc["root"]]
    while frontier:
        id = frontier.pop()
        if id in reachable:
            continue
        reachable.add(id)
        item = index.get(id)
        if item is None or item.get("visibility") not in ("public", "default", None):
            continue
        inner = item.get("inner", {})
        if "module" in inner:
            frontier.extend(inner["module"].get("items", []))
        elif "use" in inner:
            # The re-export's target, which is the id a signature will mention.
            target = inner["use"].get("id")
            if isinstance(target, int):
                reachable.add(target)
                frontier.append(target)

    def nameable(id: int) -> str | None:
        info = paths.get(id)
        if info is None or info.get("crate_id") != local:
            return FOREIGN
        return None if id in reachable else "::".join(info["path"])

    hidden: dict[str, set[str]] = {}
    for name, sig in signatures(index):
        mentioned: set[int] = set()
        type_ids(sig, mentioned)
        for id in mentioned:
            where = nameable(id)
            if where not in (None, FOREIGN):
                hidden.setdefault(where, set()).add(name)

    if hidden:
        print(f"\nUnnameable-type gate FAILED: {len(hidden)} type(s) a caller cannot write:")
        for where, users in sorted(hidden.items()):
            print(f"    {where}")
            for user in sorted(users)[:6]:
                print(f"        named by {user}")
        print("\n  Re-export the type, or make the signature take something the caller can name.")
        return 1
    print("Unnameable-type gate: OK (every public signature is writable from outside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
