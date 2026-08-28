"""Which type owns which member, read from the Rust source.

`check_api_surface.is_defined` resolved `MIPROv2::compile` by grepping for `MIPROv2` and for
`compile` *independently*, so the pair passed as long as both words appeared somewhere in 160
files. Nine types have a `compile`; deleting any one of them left every entry green. Qualifying the
ledger's names is only worth doing if something checks the pair.

The walked public surface (`rust_surface`) cannot answer this on its own: it lists a trait's method
*declaration*, not each `impl Adapter for ChatAdapter`, and 185 of the ledger's qualified names are
exactly those impls. So this reads the blocks — inherent `impl`, trait `impl`, `trait`, `struct`,
`enum` — and records what each one names.

Deliberately a scanner rather than a parser: it needs to answer "does `ChatAdapter` have a `parse`",
not to understand Rust. Where it cannot tell, it says so by returning nothing for that owner, and
the caller falls back to the older check rather than failing a name that is really there.
"""

from __future__ import annotations

import functools
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
RUST_TREES = ("crates",)

#: `impl<'a, T: Bound> Trait for Type<T>` — the header may run over several lines and end in a
#: `where` clause, so it is accumulated until the opening brace rather than matched on one line.
OPENS = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?P<kind>impl|trait|struct|enum)\b")
MEMBER = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:default\s+)?(?:const\s+|async\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*"
    r"(?:fn|const|type)\s+(?P<name>\w+)"
)
FIELD = re.compile(r"^\s*pub(?:\([^)]*\))?\s+(?P<name>\w+)\s*:")
VARIANT = re.compile(r"^\s*(?P<name>[A-Z]\w*)\s*(?:\{|\(|=|,|$)")
NAMED = re.compile(r"^(?:&(?:'\w+\s+)?(?:mut\s+)?)?(\w+)")


def _impl_target(header: str) -> tuple[str | None, str | None]:
    """`(type, trait)` an `impl` block is about. The trait is `None` for an inherent impl."""
    body = header.split(" where ", 1)[0]
    body = re.sub(r"^impl(?:\s*<.*?>)?\s+", "", body.strip(), count=1)
    trait = None
    if " for " in body:
        trait_part, body = body.split(" for ", 1)
        trait = _leading_name(trait_part)
    return _leading_name(body), trait


def _leading_name(text: str) -> str | None:
    found = NAMED.match(text.strip())
    return found.group(1) if found else None


#: A free function at module scope, which no `impl` block owns.
FREE_FN = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn (\w+)")

#: A type a file declares, for the same reason: `metric::Feedback` names one of the two.
DECLARED = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type)\s+(\w+)", re.M)


def _module_of(path: pathlib.Path) -> str:
    """The module a file defines: its stem, or its directory for a `mod.rs`."""
    return path.parent.name if path.name == "mod.rs" else path.stem


@functools.cache
def _read_all() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    owned: dict[str, set[str]] = {}
    traits: dict[str, set[str]] = {}
    for tree in RUST_TREES:
        for path in (ROOT / tree).rglob("*.rs"):
            source = path.read_text()
            _read(source, owned, traits)
            # Module-scope functions under the module's own name, so `openai::request` is a pair a
            # checker can verify. Without it a free function can only be named bare, and eight
            # files define a `request` — an entry naming one of them would pass on any of the other
            # seven, which is the whole failure the qualified names exist to stop.
            # Module-scope functions *and* the types a file declares, both under the module's own
            # name. A bare `Feedback` is two types in this crate; `metric::Feedback` is one, and a
            # module path is the only thing that separates them.
            named = {m.group(1) for m in (FREE_FN.match(line) for line in source.splitlines()) if m}
            named |= set(DECLARED.findall(source))
            if named:
                owned.setdefault(_module_of(path), set()).update(named)
            _read_nested(source, owned)
    return owned, traits


#: `pub mod roles {` — a module declared *inside* a file rather than as one.
NESTED = re.compile(r"^(?P<indent>[ 	]*)(?:pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>\w+)\s*\{")
#: What such a block declares, indented under it.
NESTED_ITEM = re.compile(
    r"^[ 	]+(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type)\s+(?P<name>\w+)"
)


def _read_nested(source: str, owned: dict[str, set[str]]) -> None:
    """Items inside a `mod name { … }` block, under that module's name.

    The file-level walk anchors its patterns at column zero, so anything a nested module declares
    was invisible: `roles::System` and `roles::Developer` are free functions inside
    `lm/api/message.rs`'s `pub mod roles`, and the ownership index knew neither. Four mapped names
    could not be qualified for that reason and no other, which is a small enough blind spot to
    close rather than to write down.

    Depth-tracked rather than indent-matched, because a nested module's body is not the only thing
    indented under it.
    """
    name: str | None = None
    depth = 0
    for line in source.splitlines():
        if name is None:
            opens = NESTED.match(line)
            if opens and "cfg(test)" not in line:
                name, depth = opens.group("name"), line.count("{") - line.count("}")
            continue
        depth += line.count("{") - line.count("}")
        found = NESTED_ITEM.match(line)
        if found:
            owned.setdefault(name, set()).add(found.group("name"))
        if depth <= 0:
            name = None


def members() -> dict[str, set[str]]:
    """`{owner: {member, ...}}` over every crate in the workspace."""
    return _read_all()[0]


def _read(source: str, owned: dict[str, set[str]], traits: dict[str, set[str]]) -> None:
    owner: str | None = None
    depth = 0
    header: str | None = None
    for line in source.splitlines():
        if owner is not None:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                owner = None
                continue
            for pattern in (MEMBER, FIELD, VARIANT):
                found = pattern.match(line)
                if found:
                    owned.setdefault(owner, set()).add(found.group("name"))
                    break
            continue
        if header is None:
            opens = OPENS.match(line)
            if not opens:
                continue
            header = line
        else:
            header += " " + line.strip()
        if "{" not in header:
            # A declaration that ends without a body — `impl Trait for Type;` cannot happen, but
            # `struct Unit;` and `trait Marker: Bound;` can, and neither owns anything.
            if header.rstrip().endswith(";"):
                header = None
            continue
        kind = OPENS.match(header).group("kind")
        if kind == "impl":
            name, trait = _impl_target(header[: header.index("{")])
            if name and trait:
                traits.setdefault(name, set()).add(trait)
        else:
            name = _leading_name(re.sub(r"^.*?\b" + kind + r"\s+", "", header, count=1))
        if name:
            owned.setdefault(name, set())
            owner, depth = name, header.count("{") - header.count("}")
        header = None


def has(owner: str, member: str) -> bool | None:
    """Whether `owner` names `member`, directly or through a trait it implements.

    A trait's *defaulted* method is reachable on every implementor without appearing in the impl
    block, so `Predict::dump_state` is real even though only `Module` writes a body for it.

    `None` where nothing was read for that owner at all, which is the caller's signal to fall back
    rather than to fail a name that is really there.
    """
    owned, traits = _read_all()
    known = owned.get(owner)
    if known is None:
        return None
    if member in known:
        return True
    return any(member in owned.get(trait, ()) for trait in traits.get(owner, ()))
