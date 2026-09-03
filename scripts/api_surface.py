"""Enumerate the public API surface dspy *defines* in the modules this crate ports.

Read from the pinned `third_party/dspy` submodule with `ast`, so the surface is deterministic — and
so `surface_of_source` can read a module out of a git object, which is how `check_pin_drift`
compares the pin against main. "Public" follows dspy's own `__all__` where a module declares one,
and otherwise every top-level class/function whose name does not start with `_`.

**Constructors are the one thing `ast` cannot answer**, so `full_surface` asks Python instead: a
class can inherit `__init__` or *be* a pydantic model, and reading the class body sees neither.
That import is confined to `full_surface`; the source-only walk keeps working without it.
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

import pinned_constructors

ROOT = pathlib.Path(__file__).parent.parent
DSPY = ROOT / "third_party" / "dspy" / "dspy"

#: The dspy modules this crate ports, relative to `third_party/dspy/dspy`. Adding a module here is
#: a claim that the crate reproduces its public API, checked against the ledger.
#:
#: A module absent here is not unexamined: `unported_modules.toml` classifies every one of them
#: with a reason, and `check_unported_modules.py` fails if the two lists are not exhaustive and
#: disjoint. That table replaced a parenthesis in this comment, which named the avatar and SIMBA
#: optimizers as unported and went on saying so after SIMBA was ported.
PORTED_MODULES = [
    "retrievers/embeddings.py",
    "clients/embedding.py",
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
    "predict/knn.py",
    # dspy.Flex. The deterministic half is ported — the signature rendering, the baseline source,
    # the vendored guest shim — and the sandbox bridge is the todo row beside it.
    "predict/flex/flex.py",
    "predict/flex/ctx.py",
    "predict/flex/bridge.py",
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
    "teleprompt/knn_fewshot.py",
    "teleprompt/vanilla.py",
    "teleprompt/copro_optimizer.py",
    "teleprompt/mipro_optimizer_v2.py",
    "teleprompt/bettertogether.py",
    "teleprompt/ensemble.py",
    "teleprompt/random_search.py",
    "teleprompt/simba.py",
    "teleprompt/simba_utils.py",
    "teleprompt/gepa/gepa.py",
    # clients (the LM stack)
    "clients/base_lm.py",
    "clients/lm.py",
    "clients/cache.py",
    "clients/provider.py",
    "clients/openai.py",
    # The canonical 3.3 wire. The OpenAI body this crate sends is byte-verified against this
    # module, and it was cited twenty-six times across the source while being on no list.
    "clients/openai_format.py",
    # evaluate
    "evaluate/evaluate.py",
    "evaluate/auto_evaluation.py",
    "evaluate/metrics.py",
    # core
    "core/types.py",
    # The tokenizer `answer_passage_match` compares a passage and an answer through. Listed with
    # that metric, which was deferred for needing it and is not any more.
    "dsp/utils/dpr.py",
    # ambient configuration — dspy.settings/configure/context are re-exports of this class's
    # methods, so the surface a user touches daily is defined here.
    "dsp/utils/settings.py",
    # Found by `check_cited_modules.py` on its first run — see `unlisted-2`. The usage tracker is
    # the one `check_coverage.py`'s docstring holds up as the lesson, whose *tests* were added when
    # that was found and whose module never was.
    "propose/propose_base.py",
    "propose/grounded_proposer.py",
    "propose/dataset_summary_generator.py",
    "propose/utils.py",
    "teleprompt/utils.py",
    "utils/parallelizer.py",
    "utils/usage_tracker.py",
    # Ported in effect and, until an audit of this list, on none: each is cited by name through the
    # source that implements it. See the `ported-in-effect-unlisted` story for how they were found.
    "teleprompt/bootstrap_trace.py",
    "teleprompt/teleprompt_optuna.py",
    "teleprompt/infer_rules.py",
    "teleprompt/gepa/instruction_proposal.py",
    "teleprompt/gepa/gepa_utils.py",
    "teleprompt/gepa/gepa_flex_utils.py",
    "utils/dummies.py",
    "primitives/python_interpreter.py",
    "utils/saving.py",
    "utils/inspect_history.py",
    "utils/hasher.py",
    "utils/mcp.py",
    # streaming. Listed because this crate ports all three and cited them by name while doing it,
    # which is the tell: a module the source explains itself in terms of is ported in effect, and
    # leaving it off this list means none of its symbols is held to the ledger.
    "streaming/streamify.py",
    "streaming/messages.py",
    "streaming/streaming_listener.py",
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


def _method_params(cls: ast.ClassDef) -> dict[str, list[str]]:
    """What each public method takes, by name — the table `_methods` does not fill.

    `_constructor_params` exists because a method list says `__init__` is there and not what it
    accepts. Every other method has the same hole, and for the teleprompters it is the larger one:
    `MIPROv2.compile` carries eighteen arguments, as much configuration as its constructor. A
    parameter that quietly has no Rust equivalent is a gap; one whose Rust spelling differs is a
    divergence; either way it is invisible to a gate that only checks the method exists.

    `__init__` is excluded — it has its own table, and listing it twice would double-count.
    `self` and `cls` are the receiver, not API.
    """
    out: dict[str, list[str]] = {}
    for stmt in cls.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if stmt.name == "__init__" or not (_public(stmt.name) or stmt.name in API_DUNDERS):
            continue
        out[stmt.name] = _parameters(stmt)
    return {name: params for name, params in out.items() if params}


def _parameters(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """A callable's public parameter names, sorted. `*args`/`**kwargs` count under their own names:
    `**kwargs` is how dspy carries a whole configuration surface, so dropping it would hide the
    thing most worth tracking."""
    args = fn.args
    named = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg is not None:
        named.append(args.vararg.arg)
    if args.kwarg is not None:
        named.append(args.kwarg.arg)
    return sorted(name for name in named if name not in ("self", "cls") and _public(name))


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
    method_params: dict[str, dict[str, list[str]]] = {}
    function_params: dict[str, list[str]] = {}
    functions: list[str] = []
    for stmt in _toplevel(tree):
        if isinstance(stmt, ast.ClassDef) and _public(stmt.name):
            classes[stmt.name] = _methods(stmt)
            params = _constructor_params(stmt)
            if params:
                constructors[stmt.name] = params
            taken = _method_params(stmt)
            if taken:
                method_params[stmt.name] = taken
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(stmt.name):
            functions.append(stmt.name)
            taken = _parameters(stmt)
            if taken:
                function_params[stmt.name] = taken
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
        method_params = {k: v for k, v in method_params.items() if k in allow}
        function_params = {k: v for k, v in function_params.items() if k in allow}
    return {
        "declares_all": declared is not None,
        "classes": classes,
        "constructors": {name: params for name, params in constructors.items() if name in classes},
        "method_params": {name: taken for name, taken in method_params.items() if name in classes},
        "function_params": {
            name: taken for name, taken in function_params.items() if name in functions
        },
        "functions": sorted(functions),
    }


def full_surface() -> dict:
    """The pinned tree's surface, with constructors answered by Python rather than by `ast`.

    Everything else is read from the source, which is what lets the same walk run over a git
    object. Constructors are the exception: a class can inherit `__init__` or *be* a pydantic
    model, and `_constructor_params` sees neither — 150 parameters a caller can pass were
    invisible to the gate, `XMLAdapter(use_native_function_calling=…)` among them. See
    `pinned_constructors`, which refuses to answer unless the installed dspy is byte-identical to
    the submodule over these same modules.
    """
    out = {}
    for rel in PORTED_MODULES:
        path = DSPY / rel
        if not path.exists():
            raise SystemExit(f"ported module missing from the pinned submodule: {rel}")
        out[rel] = surface_of(path)
    pinned_constructors.assert_pinned(PORTED_MODULES, DSPY)
    for rel, api in out.items():
        api["constructors"] = {
            name: params
            for name, params in pinned_constructors.constructors_of(
                rel, list(api["classes"]), _public
            ).items()
            if name in api["classes"]
        }
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
