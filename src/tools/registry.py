from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, Signature, _empty
from types import MethodType
from typing import Any

from smolagents import Tool
from PIL import Image


Handler = Callable[..., Any]

_INPUT_ANNOTATIONS = {
    "any": Any,
    "array": list[Any],
    "boolean": bool,
    "image": Image.Image,
    "integer": int,
    "number": float,
    "object": dict[str, Any],
    "string": str,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    inputs: dict[str, dict[str, Any]]
    output_type: str
    handler: Handler


class ToolRegistry:
    """Register plain Python handlers and build smolagents tools from them."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        inputs: dict[str, dict[str, Any]],
        output_type: str,
    ) -> Callable[[Handler], Handler]:
        def decorate(handler: Handler) -> Handler:
            if name in self._specs:
                raise ValueError(f"Tool already registered: {name}")
            self._specs[name] = ToolSpec(name, description, inputs, output_type, handler)
            return handler

        return decorate

    def build(self, context: Any = None) -> list[Tool]:
        return [_RegisteredTool(spec, context) for spec in self._specs.values()]


class _RegisteredTool(Tool):
    def __init__(self, spec: ToolSpec, context: Any):
        self.spec = spec
        self.context = context
        self.name = spec.name
        self.description = spec.description
        self.inputs = spec.inputs
        self.output_type = spec.output_type

        parameters = [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
        call_parameters = []
        annotations: dict[str, Any] = {}
        for name, schema in spec.inputs.items():
            declared_types = schema["type"]
            if isinstance(declared_types, str):
                declared_types = [declared_types]
            annotation = _INPUT_ANNOTATIONS[declared_types[0]]
            for declared_type in declared_types[1:]:
                annotation = annotation | _INPUT_ANNOTATIONS[declared_type]
            nullable = schema.get("nullable", False)
            if nullable:
                annotation = annotation | None
            annotations[name] = annotation
            default = None if nullable else _empty
            parameters.append(
                Parameter(
                    name,
                    Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
            call_parameters.append(
                Parameter(
                    name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=annotation,
                )
            )

        call_signature = Signature(call_parameters)

        def dispatch(
            _self: _RegisteredTool, *args: Any, **kwargs: Any
        ) -> Any:
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return spec.handler(context, **bound.arguments)

        forward = dispatch
        forward.__signature__ = Signature(parameters)
        forward.__annotations__ = annotations
        self.forward = MethodType(forward, self)
        super().__init__()

registry = ToolRegistry()
register_tool = registry.register


def build_tools(context: Any = None) -> list[Tool]:
    return registry.build(context)