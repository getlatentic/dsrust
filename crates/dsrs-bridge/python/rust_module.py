"""dspy modules whose orchestration is this crate's Rust, for running upstream's module tests.

Where `rust_adapter` swaps dspy's adapters and overrides `format`/`parse`, this swaps a module and
replaces the one step our Rust owns: `RustPredict` subclasses `dspy.Predict` and keeps its
`_forward_preprocess` (defaults, the input-type/extra/missing warnings, LM validation) and
`_forward_postprocess` (`Prediction.from_completions`, trace), replacing only the middle —
`adapter(lm, …) -> completions` — with a call to the crate's `Predict::forward` through
`dsrs_bridge.predict_forward`. So `test_predict` exercises our render/call/parse under dspy's real
orchestration, and everything above the wire stays dspy's own code rather than a reimplementation.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import dspy
import dspy.primitives.python_interpreter
import pydantic
from dspy.adapters.utils import format_field_value, parse_value
from dspy.primitives.code_interpreter import (
    CodeExecutionError,
    CodeInterpreterError,
    FinalOutput,
)

import crossings
import dsrs_bridge
from reflect import Unsupported, describe, described_outputs
from rust_adapter import _serialized


class RustPredict(dspy.Predict):
    """A `dspy.Predict` whose completions come from this crate's `Predict::forward`."""

    def forward(self, **kwargs):
        # dspy's own pre/post steps: defaults + warnings + LM validation, then the Prediction
        # assembly and trace. Only the completions in between are ours.
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)
        # `_forward_preprocess` has already decided whether a `prediction` kwarg is a predicted
        # output or an ordinary input. Handing it back undoes that decision, so the crate makes it
        # — otherwise Rust is a courier for an answer Python worked out, and the test would read
        # green whatever the crate did.
        predicted = config.pop("prediction", None)
        completions = self._rust_completions(lm, config, signature, demos, kwargs, predicted)
        return self._forward_postprocess(completions, signature, **kwargs)

    def _rust_completions(self, lm, config, signature, demos, inputs, predicted=None):
        # A module-level crossing: the whole render→call→parse ran in Rust.
        crossings.record_render()

        values = [
            (
                name,
                json.dumps(_serialized(inputs[name]), ensure_ascii=False),
                isinstance(inputs[name], pydantic.BaseModel),
            )
            for name in signature.input_fields
            if name in inputs
        ]
        if predicted is not None:
            values.append(("prediction", json.dumps(predicted, ensure_ascii=False), False))
        rendered_demos = [
            [
                (name, format_field_value(field_info=field, value=demo[name]))
                for name, field in {**signature.input_fields, **signature.output_fields}.items()
                if name in demo
            ]
            for demo in demos
        ]

        # Honour the adapter dspy configured — a test may set the JSON one. The Rust-backed adapters
        # carry their wire name; default to chat.
        adapter = getattr(dspy.settings.adapter, "WIRE", "chat")
        try:
            output_json, _raw = dsrs_bridge.predict_forward(
                adapter,
                signature.instructions,
                describe(signature.input_fields),
                described_outputs(signature),
                values,
                lm,
                rendered_demos or None,
                config.get("n"),
            )
        except ValueError as error:
            # A reply the crate cannot answer for — tool calls, native reasoning — carries a
            # sentinel; make it the bridge's `Unsupported` (a tracked xfail), not a hard failure.
            if "MODULE_UNSUPPORTED" in str(error):
                raise Unsupported(str(error)) from None
            raise

        # `predict_forward` returns one completion per `n`. Our parse gives JSON; dspy hands
        # `_forward_postprocess` values already coerced to their Python annotation (an enum member,
        # a `datetime`), so apply dspy's own `parse_value` per output field to restore them.
        return [
            {
                name: (
                    parse_value(value, signature.output_fields[name].annotation)
                    if name in signature.output_fields
                    else value
                )
                for name, value in fields.items()
            }
            for fields in json.loads(output_json)
        ]


class RustReAct(dspy.ReAct):
    """A `dspy.ReAct` whose loop, tool calls and extraction run in this crate's `ReAct`.

    Only `forward` is replaced; `dspy.ReAct.__init__` still builds the tool dict and the react/
    extract signatures, so `self.tools`, `self.signature` and `self.max_iters` stay dspy's. The
    crate's loop calls the same Python tools (through a `PyTool`) and the same LM (through `PyLM`).
    """

    def forward(self, interpreter=None, /, **input_args):
        # dspy 3.3.0 made the interpreter a positional-only first parameter of `forward`, so a
        # caller can hand one in and keep ownership of it. Accepted and passed through to
        # upstream's own `_interpreter_context`, which is what decides who shuts it down.
        max_iters = input_args.pop("max_iters", self.max_iters)
        lm = dspy.settings.lm
        crossings.record_render()
        values = [
            (
                name,
                json.dumps(_serialized(input_args[name]), ensure_ascii=False),
                isinstance(input_args[name], pydantic.BaseModel),
            )
            for name in self.signature.input_fields
            if name in input_args
        ]
        output_json, _raw = dsrs_bridge.react_forward(
            self.signature.instructions,
            describe(self.signature.input_fields),
            described_outputs(self.signature),
            values,
            lm,
            list(self.tools.values()),
            max_iters,
        )
        return dspy.Prediction(**json.loads(output_json))


def _input_values(signature, input_args):
    """The task's declared inputs the caller supplied, as JSON the bridge reads."""
    return [
        (name, json.dumps(_serialized(input_args[name]), ensure_ascii=False))
        for name in signature.input_fields
        if name in input_args
    ]


class RustProgramOfThought(dspy.ProgramOfThought):
    """A `dspy.ProgramOfThought` whose write/run/rewrite loop runs in this crate's module.

    Upstream's cases are `@pytest.mark.deno`, so the interpreter here is its real sandbox and the
    code the model wrote genuinely executes. What crosses is the loop: whether a snippet that
    failed is rewritten or the run gives up, and what the rewrite is told about the failure.
    """

    def forward(self, interpreter=None, /, **kwargs):
        # dspy 3.3.0 made the interpreter a positional-only first parameter of `forward`, so a
        # caller can hand one in and keep ownership of it. Accepted and passed through to
        # upstream's own `_interpreter_context`, which is what decides who shuts it down.
        # Through upstream's own `_interpreter_context`, which builds one from the factory or takes
        # the caller's and shuts down only what it built. The shim held `self.interpreter` and shut
        # it down in a `finally` — the 3.3.0b1 shape, where the module owned one for its lifetime.
        # 3.3.0 removed that attribute, so this was an AttributeError before it was a divergence.
        # Upstream's own call-time check, before any interpreter is built: `interpreter=` as a
        # keyword is a TypeError pointing at the positional form. dspy validates its caller; the
        # crate decides nothing here, so the check stays dspy's rather than being reimplemented.
        if "interpreter" in kwargs and "interpreter" not in self.signature.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument "
                "when calling the module."
            )
        with self._interpreter_context(interpreter) as repl:
            return self._run(repl, kwargs)

    def _run(self, repl, kwargs):
        # See `RustRLM.forward` for why the render is recorded here rather than at the top.
        crossings.record_render()
        try:
            # Each stage crosses as its own `_PredictorAsLM`, exactly as `RustRLM` carries
            # `generate_action` and `extract`. Upstream's tests stub these predictors one at a
            # time — `pot.code_generate = StaticPredictor(...)` — and `dspy.settings.lm` cannot
            # route to two different stubs; it was also None in every such test, so the crate's
            # loop asked nothing at all.
            output_json = dsrs_bridge.program_of_thought_forward(
                self.signature.instructions,
                describe(self.signature.input_fields),
                described_outputs(self.signature),
                _input_values(self.signature, kwargs),
                repl,
                _PredictorAsLM(self.code_generate),
                _PredictorAsLM(self.code_regenerate),
                _PredictorAsLM(self.generate_output),
                self.max_iters,
            )
        except dsrs_bridge.SandboxSessionFailed as error:
            # The interpreter's own failure, carried across as a class rather than as prose, so a
            # terminal one propagates as dspy's `CodeInterpreterError` instead of a bare
            # `ValueError`. The loops stopped feeding these back to the model; this is the last
            # step that used to flatten them again.
            raise CodeInterpreterError(str(error)) from None
        except ValueError as error:
            # dspy raises RuntimeError when the hops run out, and the message is already upstream's
            # byte for byte; only the class differs, so it is restored here.
            if str(error).startswith("Max hops reached."):
                raise RuntimeError(str(error)) from None
            raise
        return dspy.Prediction(**json.loads(output_json))


class RustCodeAct(dspy.CodeAct):
    """A `dspy.CodeAct` whose per-turn loop runs in this crate's `CodeAct`."""

    def forward(self, interpreter=None, /, **kwargs):
        # dspy 3.3.0 made the interpreter a positional-only first parameter of `forward`, so a
        # caller can hand one in and keep ownership of it. Accepted and passed through to
        # upstream's own `_interpreter_context`, which is what decides who shuts it down.
        # dspy puts the tools in the sandbox by executing their *source*, before the first turn.
        # That is upstream's setup rather than the loop under test, so it stays upstream's — and it
        # is why the crate's `define_tools` seam finds nothing left to do here.
        # See `RustProgramOfThought.forward`: the context manager owns the lifecycle now.
        # Upstream's own call-time check, before any interpreter is built: `interpreter=` as a
        # keyword is a TypeError pointing at the positional form. dspy validates its caller; the
        # crate decides nothing here, so the check stays dspy's rather than being reimplemented.
        if "interpreter" in kwargs and "interpreter" not in self.signature.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument "
                "when calling the module."
            )
        with self._interpreter_context(interpreter) as repl:
            for tool in self.tools.values():
                repl(inspect.getsource(tool.func))

            # See `RustRLM.forward` for why the render is recorded here rather than at the top.
            crossings.record_render()
            max_iters = kwargs.pop("max_iters", self.max_iters)
            # See `RustProgramOfThought`: each stage is its own `_PredictorAsLM`.
            output_json = dsrs_bridge.code_act_forward(
                self.signature.instructions,
                describe(self.signature.input_fields),
                described_outputs(self.signature),
                _input_values(self.signature, kwargs),
                repl,
                _PredictorAsLM(self.codeact),
                _PredictorAsLM(self.extractor),
                list(self.tools.values()),
                max_iters,
            )
        return dspy.Prediction(**json.loads(output_json))


class _PredictorAsLM:
    """One of RLM's two predictors, shaped like an LM so the bridge's `PyLM` can carry it.

    dspy's `test_rlm` mocks at the *predictor* level — `rlm.generate_action` is replaced by an
    object handing back canned `Prediction`s — while the crate's `Rlm` holds a `Predict` that asks
    an LM. Rather than open a second seam in the crate for a mock's benefit, the mock is dressed as
    the thing the crate already talks to: its `Prediction` is rendered back into the field blocks a
    reply arrives in, and the crate's own parse is what reads it. The loop crosses, and so does the
    parse the loop depends on.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    @property
    def field_order(self):
        """The predictor's declared outputs, where it has any — a test's mock predictor is a bare
        object with no signature at all, and answers in whatever order it likes."""
        signature = getattr(self.predictor, "signature", None)
        return list(signature.output_fields) if signature is not None else []

    def __call__(self, messages=None, n=None, **kwargs):
        lm = dspy.settings.lm
        if isinstance(self.predictor, dspy.Predict) and lm is not None:
            # A real predictor, so the crate has already rendered this turn's prompt and the model
            # answers *that*. Calling the predictor would render it a second time in Python and
            # spend a canned reply doing it — which is both a wasted response and Python answering
            # for the renderer under test.
            return lm(messages=messages, **({"n": n} if n else {}))

        # A mock predictor: it ignores what it is asked and hands back a canned `Prediction`, so
        # the reply is built back into the field blocks the crate's own parse reads.
        answered = dict(self.predictor(**kwargs).items())
        # Upstream's PoT and CodeAct stages are ChainOfThought, so the crate's ask carries a
        # `reasoning` field the *adapter* added — a stub built for upstream's loop knows nothing of
        # it, because upstream calls the predictor directly and never renders at all. Without it
        # the reply fails the marker parse, falls back to JSONAdapter, and json-repair digs the
        # first braced object out of the stub's SUBMIT text: `{'answer': 2}` presented as the
        # whole reply. Only ever the stub path — a real predictor takes the branch above.
        if "reasoning" not in answered:
            answered = {"reasoning": "stubbed", **answered}
        order = self.field_order
        names = [name for name in order if name in answered]
        names += [name for name in answered if name not in names]
        blocks = "\n\n".join(f"[[ ## {name} ## ]]\n{answered[name]}" for name in names)
        return [f"{blocks}\n\n[[ ## completed ## ]]"]


class RustRLM(dspy.RLM):
    """A `dspy.RLM` whose REPL loop runs in this crate's `Rlm`.

    `dspy.RLM.__init__` still builds both signatures and holds the interpreter, and
    `_interpreter_context` still does upstream's own per-run setup, so what the tests read off the
    object stays dspy's. Only the loop between them is ours: which turn ends the run, what lands in
    the trajectory, when the extract fallback fires, and what a submission missing a field is
    answered with — the layer no golden reaches.
    """

    def forward(self, interpreter=None, /, **input_args):
        # dspy 3.3.0 made the interpreter a positional-only first parameter of `forward`, so a
        # caller can hand one in and keep ownership of it. Accepted and passed through to
        # upstream's own `_interpreter_context`, which is what decides who shuts it down.
        #
        # `_validate_inputs` is upstream's and runs before any interpreter is built: `interpreter=`
        # as a keyword is a TypeError pointing at the positional form, and an undeclared input is a
        # ValueError. dspy validates its caller; the crate decides nothing here.
        self._validate_inputs(input_args)
        values = [
            (name, json.dumps(_serialized(input_args[name]), ensure_ascii=False))
            for name in self.signature.input_fields
            if name in input_args
        ]
        with self._interpreter_context(self._prepare_execution_tools(), interpreter) as interpreter:
            # Recorded here and not at the top of `forward`: the render count is the bytes a model
            # would read, and a call that dies in validation or in the factory renders nothing. At
            # the top it credited a crossing to `rlm(query=…)` with a factory returning None —
            # which never reached the crate — and the crossing guard read that as coverage.
            crossings.record_render()
            output_json = dsrs_bridge.rlm_forward(
                self.signature.instructions,
                describe(self.signature.input_fields),
                described_outputs(self.signature),
                values,
                interpreter,
                _PredictorAsLM(self.generate_action),
                _PredictorAsLM(self.extract),
                self.max_iters,
                self.max_llm_calls,
            )
        return dspy.Prediction(
            **{
                name: (
                    parse_value(value, self.signature.output_fields[name].annotation)
                    if name in self.signature.output_fields
                    else value
                )
                for name, value in json.loads(output_json).items()
            }
        )


#: The JSON-schema spelling of each Python type dspy's `SIMPLE_TYPES` covers. Anything else goes
#: over without a type, which is upstream's own rule: an annotation it cannot write into a
#: generated signature is dropped rather than guessed at.
_SCHEMA_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _synchronous(fn):
    """A tool the sandbox can call, whether or not the caller wrote it `async`.

    Awaiting a Python coroutine is Python's job — upstream does it in `_await_in_sync` — so the
    callable is wrapped here rather than making the Rust side reason about an event loop.
    """
    from dspy.primitives.python_interpreter import _make_jsonable

    if not inspect.iscoroutinefunction(fn):
        # dspy converts a tool's return with `_make_jsonable` before it goes back to the sandbox —
        # that is how a dataclass or a pydantic model crosses, and how a value with no JSON form
        # becomes a typed refusal instead of a serializer traceback. The shim's path bypassed
        # dspy's `_handle_tool_call`, so the conversion has to ride the wrapper.
        def converted(**kwargs):
            return _make_jsonable(fn(**kwargs))

        converted.__signature__ = inspect.signature(fn)
        return converted

    def awaited(**kwargs):
        return _make_jsonable(asyncio.run(fn(**kwargs)))

    awaited.__signature__ = inspect.signature(fn)
    return awaited


def _tool_arguments(fn):
    """One callable's arguments, in the shape `Tool::args` reads.

    Reflection over a Python signature is Python's job; what the sandbox is *told* about it is the
    crate's, in `interpreter::deno::register`. A default travels because it is what makes the
    generated `def` optional.
    """
    described = {}
    for name, parameter in inspect.signature(fn).parameters.items():
        schema = {}
        if parameter.annotation is not inspect.Parameter.empty:
            spelled = _SCHEMA_TYPES.get(parameter.annotation)
            if spelled is not None:
                schema["type"] = spelled
        if parameter.default is not inspect.Parameter.empty:
            schema["default"] = parameter.default
        described[name] = schema
    return described


class RustPythonInterpreter(dspy.primitives.python_interpreter.PythonInterpreter):
    """A `PythonInterpreter` whose `execute` is this crate's `DenoInterpreter`.

    Only that one method is replaced. Everything else — the constructor's grants, the context
    manager, `__call__` — stays dspy's own code, so a test that reaches for `deno_process` or
    `_inject_variables` is still testing Python and is declared as such.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rust = dsrs_bridge.RustSandbox(
            env=[str(v) for v in (self.enable_env_vars or [])],
            read=[str(p) for p in (self.enable_read_paths or [])],
            write=[str(p) for p in (self.enable_write_paths or [])],
            network=[str(h) for h in (self.enable_network_access or [])],
            tools=[
                (name, json.dumps(_tool_arguments(fn)), _synchronous(fn))
                for name, fn in (self.tools or {}).items()
            ],
            outputs=json.dumps(self.output_fields) if self.output_fields else None,
            sync_files=bool(self.sync_files),
        )

    def execute(self, code, variables=None):
        crossings.record_render()
        # dspy's protocol for an interpreter the caller owns: mutate `.tools` / `.output_fields`
        # and clear `_tools_registered`, and the next execute picks them up. `RLM` injects per-call
        # tools that way, and upstream's test pool configures one per test. The sandbox was told
        # only what it was built with, so a pooled interpreter kept nothing and every configured
        # tool came back as a `NameError` from inside the sandbox.
        if not getattr(self, "_tools_registered", True):
            self._rust.redefine(
                tools=[
                    (name, json.dumps(_tool_arguments(fn)), _synchronous(fn))
                    for name, fn in (self.tools or {}).items()
                ],
                outputs=json.dumps(self.output_fields) if self.output_fields else None,
            )
            self._tools_registered = True
        # dspy's own host-side conversion layer, above the interpreter protocol: `_make_jsonable`
        # is what turns a set, a dataclass, a namedtuple or a pydantic model into JSON before the
        # sandbox boundary, raising CodeInterpreterError for a value with no JSON form. This
        # override replaces dspy's `execute` whole, so skipping it here skipped the layer — and
        # `json.dumps` met a raw set. The strict xfails for `test_serialize_set` were dropped on
        # the reasoning that "dspy converts before it reaches the crate": true of dspy's own
        # interpreter, false of this shim until this line.
        from dspy.primitives.python_interpreter import _make_jsonable

        payload = json.dumps({k: _make_jsonable(v) for k, v in (variables or {}).items()})
        try:
            kind, value_json = self._rust.execute(code, payload)
        except dsrs_bridge.SandboxSessionFailed as error:
            # The interpreter's own failure — host setup, the process, the protocol. Terminal for
            # the session, and no rewrite of the submitted code repairs it.
            raise CodeInterpreterError(str(error)) from None
        except (dsrs_bridge.SandboxExecutionFailed, ValueError) as error:
            # The submitted code's failure in a healthy sandbox, which a module hands back to the
            # model to correct.
            #
            # By exception *class*, not by reading the message. The crate decides this from the
            # JSON-RPC error code and the bridge carries the variant across, because a text match
            # cannot survive `test_generated_exception_name_cannot_spoof_interpreter_failure`:
            # sandbox code declaring `class CodeInterpreterError` and raising it is still the
            # code's failure, and matching on the name reads it as the interpreter's.
            said = str(error)
            if said.startswith("Invalid Python syntax"):
                raise SyntaxError(said) from None
            raise CodeExecutionError(said) from None
        value = json.loads(value_json)
        return FinalOutput(value) if kind == "submitted" else value

    def shutdown(self):
        self._rust.shutdown()
        super().shutdown()
