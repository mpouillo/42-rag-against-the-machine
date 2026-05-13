from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolCallRequest as ToolCallRequest, ToolCallWrapper as ToolCallWrapper
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT as ContextT
from typing import Annotated, Any, Generic, Protocol, TypeAlias, overload
from typing_extensions import NotRequired, Required, TypeVar, TypedDict, Unpack

__all__ = ['AgentMiddleware', 'AgentState', 'ContextT', 'ExtendedModelResponse', 'ModelCallResult', 'ModelRequest', 'ModelResponse', 'OmitFromSchema', 'ResponseT', 'StateT_co', 'ToolCallRequest', 'ToolCallWrapper', 'after_agent', 'after_model', 'before_agent', 'before_model', 'dynamic_prompt', 'hook_config', 'wrap_tool_call']

ResponseT = TypeVar('ResponseT', default=Any)

class _ModelRequestOverrides(TypedDict, total=False):
    model: BaseChatModel
    system_message: SystemMessage | None
    messages: list[AnyMessage]
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    model_settings: dict[str, Any]
    state: AgentState[Any]

@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel
    messages: list[AnyMessage]
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)
    def __init__(self, *, model: BaseChatModel, messages: list[AnyMessage], system_message: SystemMessage | None = None, system_prompt: str | None = None, tool_choice: Any | None = None, tools: list[BaseTool | dict[str, Any]] | None = None, response_format: ResponseFormat[Any] | None = None, state: AgentState[Any] | None = None, runtime: Runtime[ContextT] | None = None, model_settings: dict[str, Any] | None = None) -> None: ...
    @property
    def system_prompt(self) -> str | None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]: ...

@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]
    structured_response: ResponseT | None = ...

@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = ...

ModelCallResult: TypeAlias

@dataclass
class OmitFromSchema:
    input: bool = ...
    output: bool = ...

class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]

class _InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]

class _OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
StateT = TypeVar('StateT', bound=AgentState[Any], default=AgentState[Any])
StateT_co = TypeVar('StateT_co', bound=AgentState[Any], default=AgentState[Any], covariant=True)
StateT_contra = TypeVar('StateT_contra', bound=AgentState[Any], contravariant=True)

class _DefaultAgentState(AgentState[Any]): ...

class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    state_schema: type[StateT]
    tools: Sequence[BaseTool]
    @property
    def name(self) -> str: ...
    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def wrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]: ...
    async def awrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]]) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]: ...
    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def aafter_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]: ...
    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]) -> ToolMessage | Command[Any]: ...

class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    def __call__(self, state: StateT_contra, runtime: Runtime[ContextT]) -> dict[str, Any] | Command[Any] | None | Awaitable[dict[str, Any] | Command[Any] | None]: ...

class _CallableReturningSystemMessage(Protocol[StateT_contra, ContextT]):
    def __call__(self, request: ModelRequest[ContextT]) -> str | SystemMessage | Awaitable[str | SystemMessage]: ...

class _CallableReturningModelResponse(Protocol[StateT_contra, ContextT, ResponseT]):
    def __call__(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage: ...

class _CallableReturningToolResponse(Protocol):
    def __call__(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]: ...
CallableT = TypeVar('CallableT', bound=Callable[..., Any])

def hook_config(*, can_jump_to: list[JumpTo] | None = None) -> Callable[[CallableT], CallableT]: ...
@overload
def before_model(func: _CallableWithStateAndRuntime[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]: ...
@overload
def before_model(func: None = None, *, state_schema: type[StateT] | None = None, tools: list[BaseTool] | None = None, can_jump_to: list[JumpTo] | None = None, name: str | None = None) -> Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...
@overload
def after_model(func: _CallableWithStateAndRuntime[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]: ...
@overload
def after_model(func: None = None, *, state_schema: type[StateT] | None = None, tools: list[BaseTool] | None = None, can_jump_to: list[JumpTo] | None = None, name: str | None = None) -> Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...
@overload
def before_agent(func: _CallableWithStateAndRuntime[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]: ...
@overload
def before_agent(func: None = None, *, state_schema: type[StateT] | None = None, tools: list[BaseTool] | None = None, can_jump_to: list[JumpTo] | None = None, name: str | None = None) -> Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...
@overload
def after_agent(func: _CallableWithStateAndRuntime[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]: ...
@overload
def after_agent(func: None = None, *, state_schema: type[StateT] | None = None, tools: list[BaseTool] | None = None, can_jump_to: list[JumpTo] | None = None, name: str | None = None) -> Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...
@overload
def dynamic_prompt(func: _CallableReturningSystemMessage[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]: ...
@overload
def dynamic_prompt(func: None = None) -> Callable[[_CallableReturningSystemMessage[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...
@overload
def wrap_tool_call(func: _CallableReturningToolResponse) -> AgentMiddleware: ...
@overload
def wrap_tool_call(func: None = None, *, tools: list[BaseTool] | None = None, name: str | None = None) -> Callable[[_CallableReturningToolResponse], AgentMiddleware]: ...
