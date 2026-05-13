from _typeshed import Incomplete
from collections.abc import Awaitable, Callable as Callable
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ModelRequest as ModelRequest, ModelResponse as ModelResponse, OmitFromInput as OmitFromInput, ResponseT as ResponseT
from langchain.tools import ToolRuntime as ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import InjectedToolCallId as InjectedToolCallId
from langgraph.runtime import Runtime as Runtime
from langgraph.types import Command
from pydantic import BaseModel
from typing import Annotated, Any, Literal
from typing_extensions import NotRequired, TypedDict, override

class Todo(TypedDict):
    content: str
    status: Literal['pending', 'in_progress', 'completed']

class PlanningState(AgentState[ResponseT]):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]

class WriteTodosInput(BaseModel):
    todos: list[Todo]

WRITE_TODOS_TOOL_DESCRIPTION: str
WRITE_TODOS_SYSTEM_PROMPT: str

def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[Any]: ...

class TodoListMiddleware(AgentMiddleware[PlanningState[ResponseT], ContextT, ResponseT]):
    state_schema = PlanningState
    system_prompt: Incomplete
    tool_description: Incomplete
    tools: Incomplete
    def __init__(self, *, system_prompt: str = ..., tool_description: str = ...) -> None: ...
    def wrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage: ...
    async def awrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]]) -> ModelResponse[ResponseT] | AIMessage: ...
    @override
    def after_model(self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    @override
    async def aafter_model(self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
