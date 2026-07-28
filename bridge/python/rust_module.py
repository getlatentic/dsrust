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
        completions = self._rust_completions(lm, config, signature, demos, kwargs)
        return self._forward_postprocess(completions, signature, **kwargs)

    def _rust_completions(self, lm, config, signature, demos, inputs):
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
