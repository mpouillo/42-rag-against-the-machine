from _typeshed import Incomplete
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, PrivateStateAttr as PrivateStateAttr, ResponseT as ResponseT, hook_config as hook_config
from langchain_core.messages import ToolCall as ToolCall
from langgraph.channels.untracked_value import UntrackedValue as UntrackedValue
from langgraph.runtime import Runtime as Runtime
from langgraph.typing import ContextT
from typing import Annotated, Any
from typing_extensions import NotRequired, override

ExitBehavior: Incomplete

class ToolCallLimitState(AgentState[ResponseT]):
    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]

class ToolCallLimitExceededError(Exception):
    thread_count: Incomplete
    run_count: Incomplete
    thread_limit: Incomplete
    run_limit: Incomplete
    tool_name: Incomplete
    def __init__(self, thread_count: int, run_count: int, thread_limit: int | None, run_limit: int | None, tool_name: str | None = None) -> None: ...

class ToolCallLimitMiddleware(AgentMiddleware[ToolCallLimitState[ResponseT], ContextT, ResponseT]):
    state_schema = ToolCallLimitState
    tool_name: Incomplete
    thread_limit: Incomplete
    run_limit: Incomplete
    exit_behavior: Incomplete
    def __init__(self, *, tool_name: str | None = None, thread_limit: int | None = None, run_limit: int | None = None, exit_behavior: ExitBehavior = 'continue') -> None: ...
    @property
    def name(self) -> str: ...
    @override
    def after_model(self, state: ToolCallLimitState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def aafter_model(self, state: ToolCallLimitState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
