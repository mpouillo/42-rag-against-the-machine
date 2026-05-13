from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelResponse, ResponseT, StateT_co, _InputAgentState, _OutputAgentState
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command
from typing import Any, Generic

__all__ = ['create_agent']

@dataclass
class _ComposedExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    commands: list[Command[Any]] = field(default_factory=list)

def create_agent(model: str | BaseChatModel, tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None, *, system_prompt: str | SystemMessage | None = None, middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (), response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None, state_schema: type[AgentState[ResponseT]] | None = None, context_schema: type[ContextT] | None = None, checkpointer: Checkpointer | None = None, store: BaseStore | None = None, interrupt_before: list[str] | None = None, interrupt_after: list[str] | None = None, debug: bool = False, name: str | None = None, cache: BaseCache[Any] | None = None, transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]: ...
