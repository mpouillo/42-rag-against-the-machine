from _typeshed import Incomplete
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, PrivateStateAttr as PrivateStateAttr, ResponseT as ResponseT, hook_config as hook_config
from langgraph.channels.untracked_value import UntrackedValue as UntrackedValue
from langgraph.runtime import Runtime as Runtime
from typing import Annotated, Any, Literal
from typing_extensions import NotRequired, override

class ModelCallLimitState(AgentState[ResponseT]):
    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]

class ModelCallLimitExceededError(Exception):
    thread_count: Incomplete
    run_count: Incomplete
    thread_limit: Incomplete
    run_limit: Incomplete
    def __init__(self, thread_count: int, run_count: int, thread_limit: int | None, run_limit: int | None) -> None: ...

class ModelCallLimitMiddleware(AgentMiddleware[ModelCallLimitState[ResponseT], ContextT, ResponseT]):
    state_schema = ModelCallLimitState
    thread_limit: Incomplete
    run_limit: Incomplete
    exit_behavior: Incomplete
    def __init__(self, *, thread_limit: int | None = None, run_limit: int | None = None, exit_behavior: Literal['end', 'error'] = 'end') -> None: ...
    @override
    def before_model(self, state: ModelCallLimitState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def abefore_model(self, state: ModelCallLimitState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    @override
    def after_model(self, state: ModelCallLimitState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def aafter_model(self, state: ModelCallLimitState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
