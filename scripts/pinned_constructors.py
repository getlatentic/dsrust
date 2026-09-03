"""What each of dspy's classes really accepts, asked of Python rather than of the source text.

`api_surface.py` reads the pinned tree with `ast`, which is right for everything it does except
this one question. A constructor is not a `def __init__` in a class body:

  - `XMLAdapter` defines none and inherits `ChatAdapter.__init__`, so
    `XMLAdapter(use_native_function_calling=True)` is real and the walk saw nothing.
  - `Code`, `History`, `LMRequest` and thirteen others are pydantic models whose constructor *is*
    their field list, and a field is an annotated assignment the walk read as a class attribute.
  - `Image` is both: a model with fields *and* an `__init__`, so it accepts the union.

Reimplementing the MRO and pydantic's field collection in `ast` would be a second implementation of
Python's own semantics, and the interesting cases are exactly the ones it would get wrong. So this
asks Python. The pin is preserved by [`assert_pinned`], which refuses to answer unless the installed
package is byte-identical to the submodule over every module we read — introspecting a *different*
dspy than the one the goldens came from would be worse than the gap it closes.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib

try:  # pydantic is a dspy dependency; absent only if the venv is not the one the gates use.
    from pydantic import BaseModel

    _PYDANTIC_INIT = BaseModel.__init__
except ImportError:  # pragma: no cover - `assert_pinned` fails first and says why.
    BaseModel = None  # type: ignore[assignment]
    _PYDANTIC_INIT = None


def module_name(rel: str) -> str:
    """`adapters/types/code.py` as `dspy.adapters.types.code`."""
    return "dspy." + rel[:-3].replace("/", ".")


def assert_pinned(modules: list[str], pinned_root: pathlib.Path) -> None:
    """Refuse to introspect an installed dspy that is not the pinned one.

    Compared as bytes over the modules actually read, not by `__version__`: a version string is
    written at build time and says nothing about whether the file in front of us is the file the
    goldens were generated from.
    """
    installed = pathlib.Path(importlib.import_module("dspy").__file__).parent
    for rel in modules:
        theirs, ours = installed / rel, pinned_root / rel
        # Both sides, because either missing means the comparison cannot be made — and a guard that
        # raises `FileNotFoundError` instead of saying which tree is wrong is not a guard.
        for label, path in (("installed dspy", theirs), ("the pinned submodule", ours)):
            if not path.exists():
                raise SystemExit(
                    f"{label} has no {rel}, so the surface cannot be checked against the pin.\n"
                    f"  run: uv sync   (and check the submodule is at its pinned commit)"
                )
        if theirs.read_bytes() != ours.read_bytes():
            raise SystemExit(
                f"installed dspy's {rel} differs from the pinned submodule.\n"
                f"  the surface would be read from a different dspy than the goldens came from.\n"
                f"  run: uv sync   (and check the submodule is at its pinned commit)"
            )


def parameters(obj: type, is_public) -> list[str]:
    """Every name `obj(...)` accepts, as Python resolves it.

    Both halves, because a class can have both: a model's fields are constructor arguments, and a
    model that also defines `__init__` accepts its named parameters *and*, through `**data`, its
    fields. `*args`/`**kwargs` are not names a caller can be said to have lost — a port either
    takes the same named argument or it does not.
    """
    names: set[str] = set()
    if BaseModel is not None and issubclass(obj, BaseModel):
        names |= set(obj.model_fields)
    init = getattr(obj, "__init__", None)
    # pydantic's generated `__init__` is `(self, **data)` and states nothing the fields do not.
    if init is not None and init is not _PYDANTIC_INIT and init is not object.__init__:
        try:
            taken = inspect.signature(init).parameters
        except (TypeError, ValueError):
            taken = {}
        names |= {
            name
            for name, param in taken.items()
            if name not in ("self", "cls")
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        }
    return sorted(name for name in names if is_public(name))


def constructors_of(rel: str, classes: list[str], is_public) -> dict[str, list[str]]:
    """`{class: [parameter, ...]}` for one module's classes, skipping what it does not define."""
    module = importlib.import_module(module_name(rel))
    out: dict[str, list[str]] = {}
    for name in classes:
        obj = getattr(module, name, None)
        if not inspect.isclass(obj):
            continue
        taken = parameters(obj, is_public)
        if taken:
            out[name] = taken
    return out
