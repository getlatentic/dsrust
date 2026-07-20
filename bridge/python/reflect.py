"""What a dspy signature looks like to the crate: one plain description per field.

Reading a type off a Python annotation is reflection, and only Python can do it. Deciding how
the result reads in a prompt is rendering, and only the crate does that. This module is the
line between: every function here answers a question about an annotation and answers it in
data, so nothing on this side ever formats a byte the model will see.

A question this side cannot answer is raised rather than guessed at. `Unsupported` derives from
`BaseException` so it lands outside every `except Exception` dspy runs, because a case rendered
on Python would report a pass for code this crate has not written.
"""

from __future__ import annotations

import enum
import json
import types
import typing

import pydantic
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.code import Code
from dspy.adapters.utils import (
    _annotation_is_subclass,
    _get_json_schema,
    get_annotation_name,
)

# The Rust FieldKind each Python annotation maps to. Anything absent is not yet modelled.
KINDS: dict[typing.Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}


class Unsupported(BaseException):
    """The case needs a Rust feature that does not exist yet.

    Deriving from `BaseException` puts this outside `except Exception`, which is what dspy's
    JSON-adapter fallback catches on both paths. Inheriting from `Exception` would leave the
    exclusion resting on a matching `except Unsupported: raise` in every handler that could
    ever see one, and the async path already inherits a handler this shim does not write.
    """


def kind_of(annotation: typing.Any) -> str:
    """A scalar names itself; anything else carries the name dspy would print.

    Sending `json:<annotation>` rather than a bare `json` is what lets the numbered line read
    `(PetOwner)` the way dspy renders it, instead of collapsing every non-scalar to one word.
    """
    if typing.get_origin(annotation) is typing.Literal:
        # The closed set travelling beside this becomes the printed annotation, so what is
        # named here is the wire type under it: text, which is how every member of a Literal
        # reaches the marker path whatever its Python type.
        return "str"
    try:
        return KINDS[annotation]
    except (KeyError, TypeError):
        pass
    # Ahead of the JSON kinds: pydantic can schema an enum, but dspy does not describe one that
    # way — it names the type and lists its members' values.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return f"enum:{get_annotation_name(annotation)}"
    if _carries_as_json(annotation):
        return f"json:{get_annotation_name(annotation)}"
    raise Unsupported(f"no Rust FieldKind for annotation {annotation!r}")


def _carries_as_json(annotation: typing.Any) -> bool:
    """Whether the crate's `Json` kind can carry values of this annotation.

    A model does, since its values are objects. A container does exactly when everything it
    holds does — `list[Tool]` rides on `Tool` — because the container is JSON either way and
    what is inside it is what has to survive the crossing. Anything else says so rather than
    crossing as an annotation whose values were never checked: a kind that renders a value
    wrongly is worse than one that refuses it, because the prompt still looks plausible.
    """
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return True
    # A plain class pydantic can describe carries too — `datetime` reaches the model as the
    # string its schema says it is. Asking pydantic is the same question the crate's schema
    # note asks later, so a type that answers it renders consistently.
    if isinstance(annotation, type) and not typing.get_args(annotation):
        try:
            pydantic.TypeAdapter(annotation).json_schema()
            return True
        except Exception:
            return False
    args = typing.get_args(annotation)
    return bool(args) and all(
        arg is Ellipsis or arg is type(None) or _scalar_or_json(arg) for arg in args
    )


def _scalar_or_json(annotation: typing.Any) -> bool:
    """Whether one member of a container is itself carryable."""
    try:
        return annotation in KINDS or _carries_as_json(annotation)
    except TypeError:
        return _carries_as_json(annotation)


def closed_set_of(annotation: typing.Any) -> str | None:
    """A `Literal`'s members as JSON, or None where the annotation is not one.

    JSON is what the bridge carries a closed set over, and it spells the three member types
    Rust models. A `Literal` over anything else — an Enum, None, bytes — has no crossing yet,
    and must say so rather than lose members on the way across.
    """
    # An enum's members are its closed set, carried as the values dspy asks the model for.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return json.dumps([member.value for member in annotation])
    if typing.get_origin(annotation) is not typing.Literal:
        return None
    members = typing.get_args(annotation)
    # A member Python prints as itself — an enum member — crosses as its `str`, tagged so the
    # crate spells it bare rather than quoting it into something the model would answer wrong.
    return json.dumps(
        [
            member if isinstance(member, (str, int, bool)) else {"bare": str(member)}
            for member in members
        ]
    )


def schema_of(kind: str, annotation: typing.Any) -> str | None:
    """A structured field's JSON schema, as dspy builds it, or None for a scalar.

    Reading a schema off a Python annotation is pydantic's job, so upstream's own extractor runs
    here — key order included, since it is part of the bytes. Whether the schema reaches the
    prompt stays the crate's decision: a type whose description already states its contract
    drops the note, and the crate is what knows that.

    Only a structured field is asked for one, which is where the crate consults it and where
    dspy computes it. An annotation that cannot produce one is a gap in this bridge rather than
    a field to render blank, so it says so instead of quietly dropping the schema — a missing
    note renders a prompt that looks right and is not.
    """
    if not kind.startswith("json:"):
        return None
    try:
        return json.dumps(_get_json_schema(annotation), ensure_ascii=False)
    except Exception as error:
        raise Unsupported(f"no JSON schema for annotation {annotation!r}: {error}") from error


def type_descriptions_of(annotation: typing.Any) -> str | None:
    """The custom types an annotation names, as JSON `[[name, prose], ...]`, or None.

    Which types an annotation mentions is Python reflection, so it happens here; how the pairs
    read in a prompt is the crate's business, so only the pairs cross.
    """
    described = [
        {
            "name": get_annotation_name(custom),
            "text": custom.description(),
            # dspy asks whether the annotation *is* a `dspy.Code`, not what it is called, so
            # this asks the same way: a subclass counts and a look-alike does not.
            "replaces_schema": _annotation_is_subclass(custom, Code),
        }
        for custom in Type.extract_custom_type_from_annotation(annotation)
        if custom.description()
    ]
    return json.dumps(described) if described else None


def reflection_of(kind: str, annotation: typing.Any) -> str | None:
    """A structured field's own shape, as JSON, or None for a scalar.

    An adapter that states the declared type itself rather than a schema of it needs facts a
    schema throws away: a member keyed by its name *and* the alias it answers to elsewhere,
    the difference between `object` and a type with no constraints, a mapping's key type, and
    every model's docstring. Those come off pydantic, so they are read here, as the tree dspy's
    own renderer walks — with nothing walked into text, which is the crate's half.

    Models cross as a table referred to by index, so a type that names itself arrives whole
    instead of recursing forever on this side. Whether such a type can be rendered at all is
    then the crate's call, made where dspy makes it.
    """
    if not kind.startswith("json:"):
        return None
    models: list = []
    seen: dict[type, int] = {}
    return json.dumps({"type": _node(annotation, models, seen), "models": models})


def _node(annotation: typing.Any, models: list, seen: dict) -> dict:
    """One node of an annotation, in the same order of questions dspy's renderer asks them."""
    for scalar, kind in KINDS.items():
        if annotation is scalar:
            return {"kind": kind}
    if _annotation_is_subclass(annotation, pydantic.BaseModel):
        return {"kind": "model", "model": _model(annotation, models, seen)}
    try:
        origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    except Exception:
        return _named(annotation)
    if origin in (types.UnionType, typing.Union):
        present = [arg for arg in args if arg is not type(None)]
        return {
            "kind": "union",
            "of": [_node(arg, models, seen) for arg in present],
            "optional": len(present) < len(args),
        }
    if origin is typing.Literal:
        return _literal(annotation, args)
    if origin is list:
        return {"kind": "list", "of": _node(args[0], models, seen)}
    if origin is dict:
        return {
            "kind": "dict",
            "key": _node(args[0], models, seen),
            "value": _node(args[1], models, seen),
        }
    return _named(annotation)


def _model(model: type, models: list, seen: dict) -> int:
    """This model's slot in the table, filling it the first time the model is reached.

    The slot is claimed before the members are walked, so a model that names itself refers back
    to the entry being built rather than recursing here forever.
    """
    if model in seen:
        return seen[model]
    index = seen[model] = len(models)
    models.append(None)
    models[index] = {
        "doc": model.__doc__,
        "fields": [
            {
                "name": name,
                "desc": field.description,
                "alias": field.alias,
                "type": _node(field.annotation, models, seen),
            }
            for name, field in model.model_fields.items()
        ],
    }
    return index


def _literal(annotation: typing.Any, members: tuple) -> dict:
    """A closed set as its members, which the crate spells the way Python's own `str` does.

    A member JSON has no spelling for — an Enum, bytes — would have to cross as text this side
    had already formatted, so the whole type falls back to its printed name instead.
    """
    if all(member is None or isinstance(member, (str, int, bool)) for member in members):
        return {"kind": "literal", "members": list(members)}
    return _named(annotation)


def _named(annotation: typing.Any) -> dict:
    """dspy's last resort, and this one: the name Python prints for the type."""
    name = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
    return {"kind": "named", "name": name}


def describe(fields: dict) -> list[tuple]:
    described = []
    for name, info in fields.items():
        desc = info.json_schema_extra.get("desc") or ""
        if desc == f"${{{name}}}":  # dspy's placeholder for "no description given"
            desc = ""
        annotation = info.annotation
        kind = kind_of(annotation)
        described.append(
            (
                name,
                kind,
                desc,
                closed_set_of(annotation),
                type_descriptions_of(annotation),
                reflection_of(kind, annotation),
            )
        )
    return described


def described_outputs(signature) -> list[tuple]:
    """Outputs carry the nested schema of a structured field ahead of the closed set."""
    return [
        (
            name,
            kind,
            desc,
            schema_of(kind, signature.output_fields[name].annotation),
            values,
            types_named,
            reflection,
        )
        for name, kind, desc, values, types_named, reflection in describe(signature.output_fields)
    ]
