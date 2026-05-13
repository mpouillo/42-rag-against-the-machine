from _typeshed import Incomplete
from collections.abc import Awaitable, Callable as Callable
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ToolCallRequest as ToolCallRequest
from langchain.chat_models.base import init_chat_model as init_chat_model
from langchain.tools import BaseTool as BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.types import Command as Command
from typing import Any, Generic

class LLMToolEmulator(AgentMiddleware[AgentState[Any], ContextT], Generic[ContextT]):
    emulate_all: Incomplete
    tools_to_emulate: set[str]
    model: Incomplete
    def __init__(self, *, tools: list[str | BaseTool] | None = None, model: str | BaseChatModel | None = None) -> None: ...
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]: ...
    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]) -> ToolMessage | Command[Any]: ...
