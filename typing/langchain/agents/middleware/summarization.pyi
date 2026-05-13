from _typeshed import Incomplete
from collections.abc import Callable, Iterable
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ResponseT as ResponseT
from langchain.chat_models import BaseChatModel as BaseChatModel, init_chat_model as init_chat_model
from langchain_core.messages import AnyMessage as AnyMessage, MessageLikeRepresentation
from langgraph.runtime import Runtime as Runtime
from typing import Any
from typing_extensions import override

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]
DEFAULT_SUMMARY_PROMPT: str
ContextFraction: Incomplete
ContextTokens: Incomplete
ContextMessages: Incomplete
ContextSize = ContextFraction | ContextTokens | ContextMessages

class SummarizationMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    model: Incomplete
    trigger: ContextSize | list[ContextSize] | None
    keep: Incomplete
    token_counter: Incomplete
    summary_prompt: Incomplete
    trim_tokens_to_summarize: Incomplete
    def __init__(self, model: str | BaseChatModel, *, trigger: ContextSize | list[ContextSize] | None = None, keep: ContextSize = ..., token_counter: TokenCounter = ..., summary_prompt: str = ..., trim_tokens_to_summarize: int | None = ..., **deprecated_kwargs: Any) -> None: ...
    @override
    def before_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    @override
    async def abefore_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
