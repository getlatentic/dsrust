"""Every Rust block in the guides, extracted and made into compilable Rust.

The guides are the first thing a caller reads and the last thing anything checks. Both
`check_external_consumer.sh` and `tests/every_spelling.rs` claimed to hold them, and both held a
*transcription* instead: they were written alongside the docs and drifted apart from them. The
README told a reader to write `use dsrust::{call, predict};` for four commits after `predict!`
became `Predict!`, while the gate that exists to compile the README's opening program compiled the
corrected import and passed.

So this reads the markdown itself. There is nothing to keep in sync.

A guide is prose, not a program, and three things follow from that:

* **A fragment names things the page never declares** — `trainset`, `Haiku`, `metric`. They come
  from `docs_fixtures.rs`, which is the only place a name may be supplied from outside the page.
* **`…` stands for code a reader is meant to fill in.** It becomes `todo!()`, which type-checks as
  anything, so `Ok(out) => …` compiles as the guide's own control flow. Only outside string
  literals: `question = "…"` is a question with an ellipsis in it.
* **Blocks share one scope, in reading order.** A page declares `Outline` in one block and calls it
  in the next, so each generated file is the page read top to bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "docs_fixtures.rs"

GUIDES = ("README.md", "docs/usage.md")

# What a guide's prose introduces before a fragment uses it: "your own metric", "a trainset", "the
# module you are wrapping". Spliced into every statement fragment as one destructuring `let`, because
# a `let` inside a `macro_rules!` is hygienic and would not be visible to the block.
#
# The list lives here rather than in docs_fixtures.rs because it is part of how a fragment is
# wrapped; the types it names are declared there. Adding to it makes the guides *less* checked, so a
# new entry needs to be a thing the page genuinely leaves to prose.
PROSE = (
    "trainset",
    "valset",
    "metric",
    "reflection_lm",
    "tools",
    "extractor",
    "inputs",
    "program",
    "qa",
    "f",
    "url",
)

FENCE = re.compile(r"^```(\w[\w,]*)\s*$")

# A block whose first meaningful line opens one of these declares something the rest of the page
# may use, so it belongs at the top level of the file rather than inside a function.
ITEM_OPENERS = ("#[derive", "#[tokio", "struct ", "enum ", "impl ", "fn ", "pub fn ", "use ", "trait ")


@dataclass(frozen=True)
class Snippet:
    """One fenced block, with where it came from — a compile error has to point back at the page."""

    guide: str
    line: int
    tags: frozenset[str]
    code: str

    @property
    def is_program(self) -> bool:
        return "fn main" in self.code

    @property
    def is_items(self) -> bool:
        """Whether the whole block declares things, so a later block on the page can use them.

        A block that mixes the two — a named reward function and then the three lines using it — is
        held as statements, where a nested `fn` is legal and the declaration is simply local.
        """
        meaningful = [line for line in self.code.splitlines() if line.strip() and not line.startswith("//")]
        if any(line.startswith("let ") for line in meaningful):
            return False
        return bool(meaningful) and meaningful[0].startswith(ITEM_OPENERS)

    @property
    def ident(self) -> str:
        return f"{Path(self.guide).stem.replace('-', '_')}_line_{self.line}"


def extract(guide: str) -> list[Snippet]:
    """Every ```rust block in one guide, in reading order."""
    text = (ROOT / guide).read_text()
    snippets: list[Snippet] = []
    tags: frozenset[str] | None = None
    body: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if tags is None:
            if match := FENCE.match(line):
                info = match.group(1).split(",")
                if info[0] == "rust":
                    tags, body, start = frozenset(info[1:]), [], number + 1
            continue
        if line.rstrip() == "```":
            snippets.append(Snippet(guide, start, tags, "\n".join(body)))
            tags = None
            continue
        body.append(line)
    return snippets


def fill_ellipses(code: str) -> str:
    """`…` becomes `todo!()`, except inside a string literal where it is prose."""
    out, in_string, escaped = [], False, False
    for character in code:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            out.append(character)
            continue
        if character == '"':
            in_string = True
            out.append(character)
        elif character == "…":
            out.append("todo!()")
        else:
            out.append(character)
    return "".join(out)


def as_rust(snippet: Snippet) -> str:
    """One block as top-level Rust: items as they are, statements inside an uncalled function."""
    code = fill_ellipses(snippet.code)
    where = f"// {snippet.guide}:{snippet.line}"
    if snippet.is_items:
        return f"{where}\n{code}\n"
    # `.await?` needs a fallible async body, and the terminating `;` goes on its own line so a
    # fragment ending in an expression, a semicolon, a `}` or a trailing comment all read alike — a
    # guide ends a fragment wherever its sentence ends. Never called: a fragment is held to
    # type-checking, not to reaching a provider.
    bindings = ", ".join(f"mut {name}" for name in PROSE)
    return (
        f"{where}\nasync fn {snippet.ident}() -> anyhow::Result<()> {{\n"
        f"    let Prose {{ {bindings} }} = prose();\n"
        f"    #[allow(redundant_semicolons, unused_must_use, path_statements)]\n"
        f"    {{\n{code}\n;\n    }}\n    Ok(())\n}}\n"
    )


def generate(guide: str) -> tuple[str, list[str]]:
    """One guide as one Rust file, plus the whole programs it contains as their own."""
    snippets = [s for s in extract(guide) if "ignore" not in s.tags]
    fragments = [as_rust(s) for s in snippets if not s.is_program]
    programs = [fill_ellipses(s.code) for s in snippets if s.is_program]
    header = f"//! Generated by scripts/docs_snippets.py from {guide}. Edit the guide.\n"
    return header + FIXTURES.read_text() + "\n" + "\n".join(fragments), programs
