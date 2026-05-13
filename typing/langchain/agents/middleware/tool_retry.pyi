from _typeshed import Incomplete
from collections.abc import Awaitable, Callable as Callable
from langchain.agents.middleware._retry import OnFailure as OnFailure, RetryOn as RetryOn, calculate_delay as calculate_delay, should_retry_exception as should_retry_exception, validate_retry_params as validate_retry_params
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ResponseT as ResponseT, ToolCallRequest as ToolCallRequest
from langchain.tools import BaseTool as BaseTool
from langchain_core.messages import ToolMessage
from langgraph.types import Command as Command
from typing import Any

class ToolRetryMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    max_retries: Incomplete
    tools: Incomplete
    retry_on: Incomplete
    on_failure: Incomplete
    backoff_factor: Incomplete
    initial_delay: Incomplete
    max_delay: Incomplete
    jitter: Incomplete
    def __init__(self, *, max_retries: int = 2, tools: list[BaseTool | str] | None = None, retry_on: RetryOn = ..., on_failure: OnFailure = 'continue', backoff_factor: float = 2.0, initial_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True) -> None: ...
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]: ...
    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]) -> ToolMessage | Command[Any]: ...
