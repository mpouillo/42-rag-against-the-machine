from collections.abc import Awaitable, Callable as Callable
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ModelRequest as ModelRequest, ModelResponse as ModelResponse, ResponseT as ResponseT
from langchain.chat_models import init_chat_model as init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel as BaseChatModel
from langchain_core.messages import AIMessage as AIMessage

class ModelFallbackMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    models: list[BaseChatModel]
    def __init__(self, first_model: str | BaseChatModel, *additional_models: str | BaseChatModel) -> None: ...
    def wrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage: ...
    async def awrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]]) -> ModelResponse[ResponseT] | AIMessage: ...
