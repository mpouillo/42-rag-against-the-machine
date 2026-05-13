from _typeshed import Incomplete
from collections.abc import Callable as Callable
from dataclasses import dataclass
from langchain_core.messages import AIMessage as AIMessage
from langchain_core.tools import BaseTool as BaseTool
from types import UnionType
from typing import Any, Generic, TypeVar
from typing_extensions import Self

SchemaT = TypeVar('SchemaT')
SchemaKind: Incomplete

class StructuredOutputError(Exception):
    ai_message: AIMessage

class MultipleStructuredOutputsError(StructuredOutputError):
    tool_names: Incomplete
    ai_message: Incomplete
    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None: ...

class StructuredOutputValidationError(StructuredOutputError):
    tool_name: Incomplete
    source: Incomplete
    ai_message: Incomplete
    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None: ...

@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    name: str
    description: str
    schema_kind: SchemaKind
    json_schema: dict[str, Any]
    strict: bool | None = ...
    def __init__(self, schema: type[SchemaT] | dict[str, Any], *, name: str | None = None, description: str | None = None, strict: bool | None = None) -> None: ...

@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT] | UnionType | dict[str, Any]
    schema_specs: list[_SchemaSpec[Any]]
    tool_message_content: str | None
    handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    def __init__(self, schema: type[SchemaT] | UnionType | dict[str, Any], *, tool_message_content: str | None = None, handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True) -> None: ...

@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_spec: _SchemaSpec[SchemaT]
    def __init__(self, schema: type[SchemaT] | dict[str, Any], *, strict: bool | None = None) -> None: ...
    def to_model_kwargs(self) -> dict[str, Any]: ...

@dataclass
class OutputToolBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind
    tool: BaseTool
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self: ...
    def parse(self, tool_args: dict[str, Any]) -> SchemaT: ...

@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind
    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self: ...
    def parse(self, response: AIMessage) -> SchemaT: ...

class AutoStrategy(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    def __init__(self, schema: type[SchemaT] | dict[str, Any]) -> None: ...
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
