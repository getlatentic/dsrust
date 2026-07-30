"""Enumerate the public API surface dspy *defines* in the modules this crate ports.

Read from the pinned `third_party/dspy` submodule with `ast`, so the surface is deterministic and
needs no import of dspy or its dependencies. "Public" follows dspy's own `__all__` where a module
declares one, and otherwise every top-level class/function whose name does not start with `_`.
For a public class, its public methods count too — `Predict` missing `forward` is exactly the kind
of API gap this measures — plus `__call__`, the invocation entry point.

This is the raw material for the ledger: every symbol it lists must be mapped to a Rust counterpart
or justified as an intended divergence in `api_ledger.toml`, and `check_api_surface.py` fails when
the two part company.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
DSPY = ROOT / "third_party" / "dspy" / "dspy"

#: The dspy modules this crate ports, relative to `third_party/dspy/dspy`. A module absent here is
#: out of scope for API conformance — either not ported (KNN, the avatar/simba
#: optimizers) or infrastructure with no public surface to mirror. Adding a module here is a claim
#: that the crate reproduces its public API, checked against the ledger.
PORTED_MODULES = [
    # adapters
    "adapters/base.py",
    "adapters/chat_adapter.py",
    "adapters/json_adapter.py",
    "adapters/xml_adapter.py",
    "adapters/baml_adapter.py",
    "adapters/two_step_adapter.py",
    "adapters/utils.py",
    "adapters/types/base_type.py",
    "adapters/types/image.py",
    "adapters/types/audio.py",
    "adapters/types/file.py",
    "adapters/types/code.py",
    "adapters/types/citation.py",
    "adapters/types/document.py",
    "adapters/types/history.py",
    "adapters/types/reasoning.py",
    "adapters/types/tool.py",
    # signatures
    "signatures/signature.py",
    "signatures/field.py",
    "signatures/utils.py",
    # predict
    "predict/predict.py",
    "predict/parameter.py",
    "predict/chain_of_thought.py",
    "predict/react.py",
    "predict/react_v2.py",
    "predict/refine.py",
    "predict/best_of_n.py",
    "predict/parallel.py",
    "predict/program_of_thought.py",
    "predict/code_act.py",
    "predict/rlm.py",
    "predict/multi_chain_comparison.py",
    "predict/aggregation.py",
    # primitives
    "primitives/example.py",
    "primitives/prediction.py",
    "primitives/module.py",
    "primitives/base_module.py",
    "primitives/code_interpreter.py",
    "primitives/repl_types.py",
    "primitives/sandbox_serializable.py",
    # the error taxonomy: 17 public types dspy exports and modules branch on
    "utils/exceptions.py",
    # the callback protocol: a base class of no-op handlers, which is a Rust trait
    "utils/callback.py",
    # teleprompt
    "teleprompt/teleprompt.py",
    "teleprompt/bootstrap.py",
    "teleprompt/vanilla.py",
    "teleprompt/copro_optimizer.py",
    "teleprompt/mipro_optimizer_v2.py",
    "teleprompt/bettertogether.py",
    "teleprompt/ensemble.py",
    "teleprompt/random_search.py",
    "teleprompt/gepa/gepa.py",
    # clients (the LM stack)
    "clients/base_lm.py",
    "clients/lm.py",
    "clients/cache.py",
    "clients/provider.py",
    "clients/openai.py",
    # evaluate
    "evaluate/evaluate.py",
    "evaluate/metrics.py",
    # core
    "core/types.py",
    # ambient configuration — dspy.settings/configure/context are re-exports of this class's
    # methods, so the surface a user touches daily is defined here.
    "dsp/utils/settings.py",
]

#: Dunder methods that are genuine API — the model is called, not just constructed — so they count
#: while the rest of the dunder protocol (`__repr__`, `__eq__`, …) does not.
API_DUNDERS = {"__call__"}


def _literal_all(node: ast.Module) -> list[str] | None:
    """The names in a module-level `__all__ = [...]`, or None if it declares none literally."""
    for stmt in node.body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            if isinstance(stmt.value, (ast.List, ast.Tuple)):
                return [e.value for e in stmt.value.elts if isinstance(e, ast.Constant)]
    return None


def _public(name: str) -> bool:
    return not name.startswith("_")


def _methods(cls: ast.ClassDef) -> list[str]:
    """A class's public methods, plus any API dunder it defines."""
    out = []
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(stmt.name) or stmt.name in API_DUNDERS:
                out.append(stmt.name)
    return sorted(out)


def _constructor_params(cls: ast.ClassDef) -> list[str]:
    """What `__init__` takes, by name.

    A method list says `__init__` exists. It does not say what it accepts, and that is where a
    port loses things quietly: `dspy.LM(model, temperature=…, max_tokens=…)` kept both on the
    instance and merged them into every call, while this crate had neither and the gate read
    216/216. The parameter names are the API as much as the method names are.
    """
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
            args = stmt.args
            named = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
            return sorted(name for name in named if name != "self" and _public(name))
    return []


def _toplevel(node: ast.Module):
    """Top-level statements, descending one level into conditional blocks that guard defs."""
    for stmt in node.body:
        yield stmt
        if isinstance(stmt, (ast.If, ast.Try)):
            for inner in getattr(stmt, "body", []):
                yield inner


def surface_of(path: pathlib.Path) -> dict:
    return surface_of_source(path.read_text(), str(path))


def surface_of_source(source: str, where: str = "<pinned>") -> dict:
    """The same, from text — so a module can be read out of a git object rather than the worktree."""
    tree = ast.parse(source, filename=where)
    declared = _literal_all(tree)
    classes: dict[str, list[str]] = {}
    constructors: dict[str, list[str]] = {}
    functions: list[str] = []
    for stmt in _toplevel(tree):
        if isinstance(stmt, ast.ClassDef) and _public(stmt.name):
            classes[stmt.name] = _methods(stmt)
            params = _constructor_params(stmt)
            if params:
                constructors[stmt.name] = params
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(stmt.name):
            functions.append(stmt.name)
        elif declared is not None:
            # A name bound to a type rather than defined as one: `LMPart = Annotated[Union[...]]`,
            # `ToolCall = LMToolCallPart`. Walking only classes and functions missed both, and both
            # are named in `core/types.py`'s `__all__`.
            #
            # Only where a module declares `__all__`, and the filter below keeps only what it names —
            # otherwise every module-level constant and `logger` would read as public surface.
            functions.extend(
                target.id
                for target in getattr(stmt, "targets", [])
                if isinstance(target, ast.Name) and _public(target.id)
            )
    # `__all__` is dspy's own word on what is public: keep only what it names, but never drop a
    # class's methods — those are not top-level names it would list.
    if declared is not None:
        allow = set(declared)
        classes = {k: v for k, v in classes.items() if k in allow}
        functions = [f for f in functions if f in allow]
    return {
        "declares_all": declared is not None,
        "classes": classes,
        "constructors": {name: params for name, params in constructors.items() if name in classes},
        "functions": sorted(functions),
    }


def full_surface() -> dict:
    out = {}
    for rel in PORTED_MODULES:
        path = DSPY / rel
        if not path.exists():
            raise SystemExit(f"ported module missing from the pinned submodule: {rel}")
        out[rel] = surface_of(path)
    return out


def symbol_keys(surface: dict) -> set[str]:
    """Every symbol as a flat `module::name` (class), `module::name` (function), or
    `module::Class.method` key — the identity the ledger is keyed by."""
    keys = set()
    for module, api in surface.items():
        for cls, methods in api["classes"].items():
            keys.add(f"{module}::{cls}")
            for method in methods:
                keys.add(f"{module}::{cls}.{method}")
        for fn in api["functions"]:
            keys.add(f"{module}::{fn}")
    return keys


def main() -> None:
    surface = full_surface()
    if "--json" in sys.argv:
        print(json.dumps(surface, indent=2))
        return
    keys = symbol_keys(surface)
    classes = sum(len(a["classes"]) for a in surface.values())
    methods = sum(len(m) for a in surface.values() for m in a["classes"].values())
    functions = sum(len(a["functions"]) for a in surface.values())
    print(f"ported modules : {len(surface)}")
    print(f"classes        : {classes}")
    print(f"methods        : {methods}")
    print(f"functions      : {functions}")
    print(f"total symbols  : {len(keys)}")


if __name__ == "__main__":
    main()
