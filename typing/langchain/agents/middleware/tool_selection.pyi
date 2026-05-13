from _typeshed import Incomplete
from collections.abc import Awaitable, Callable as Callable
from dataclasses import dataclass
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ModelRequest as ModelRequest, ModelResponse as ModelResponse, ResponseT as ResponseT
from langchain.chat_models.base import init_chat_model as init_chat_model
from langchain.tools import BaseTool as BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage as AIMessage, HumanMessage

logger: Incomplete
DEFAULT_SYSTEM_PROMPT: str

@dataclass
class _SelectionRequest:
    available_tools: list[BaseTool]
    system_message: str
    last_user_message: HumanMessage
    model: BaseChatModel
    valid_tool_names: list[str]

class LLMToolSelectorMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    system_prompt: Incomplete
    max_tools: Incomplete
    always_include: Incomplete
    model: BaseChatModel | None
    def __init__(self, *, model: str | BaseChatModel | None = None, system_prompt: str = ..., max_tools: int | None = None, always_include: list[str] | None = None) -> None: ...
    def wrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage: ...
    async def awrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]]) -> ModelResponse[ResponseT] | AIMessage: ...
