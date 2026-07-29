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

import inspect
import json

import dspy
import pydantic
from dspy.adapters.utils import format_field_value, parse_value

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

    def forward(self, **input_args):
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

    def forward(self, **kwargs):
        crossings.record_render()
        try:
            output_json = dsrs_bridge.program_of_thought_forward(
                self.signature.instructions,
                describe(self.signature.input_fields),
                described_outputs(self.signature),
                _input_values(self.signature, kwargs),
                self.interpreter,
                dspy.settings.lm,
                self.max_iters,
            )
        except ValueError as error:
            # The crate answers with an untyped `anyhow::Error`, which the bridge can only hand
            # over as a ValueError, and dspy raises RuntimeError when the hops run out. The message
            # is already upstream's byte for byte; only the class differs, so it is restored here.
            # The real fix is the crate's error taxonomy (#10), after which this can read a type.
            if str(error).startswith("Max hops reached."):
                raise RuntimeError(str(error)) from None
            raise
        finally:
            self.interpreter.shutdown()
        return dspy.Prediction(**json.loads(output_json))


class RustCodeAct(dspy.CodeAct):
    """A `dspy.CodeAct` whose per-turn loop runs in this crate's `CodeAct`."""

    def forward(self, **kwargs):
        # dspy puts the tools in the sandbox by executing their *source*, before the first turn.
        # That is upstream's setup rather than the loop under test, so it stays upstream's — and it
        # is why the crate's `define_tools` seam finds nothing left to do here.
        for tool in self.tools.values():
            self.interpreter(inspect.getsource(tool.func))

        crossings.record_render()
        max_iters = kwargs.pop("max_iters", self.max_iters)
        output_json = dsrs_bridge.code_act_forward(
            self.signature.instructions,
            describe(self.signature.input_fields),
            described_outputs(self.signature),
            _input_values(self.signature, kwargs),
            self.interpreter,
            dspy.settings.lm,
            list(self.tools.values()),
            max_iters,
        )
        self.interpreter.shutdown()
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

    def forward(self, **input_args):
        crossings.record_render()
        values = [
            (name, json.dumps(_serialized(input_args[name]), ensure_ascii=False))
            for name in self.signature.input_fields
            if name in input_args
        ]
        with self._interpreter_context(self._prepare_execution_tools()) as interpreter:
            output_json = dsrs_bridge.rlm_forward(
                self.signature.instructions,
                describe(self.signature.input_fields),
                described_outputs(self.signature),
                values,
                interpreter,
                _PredictorAsLM(self.generate_action),
                _PredictorAsLM(self.extract),
                self.max_iterations,
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
