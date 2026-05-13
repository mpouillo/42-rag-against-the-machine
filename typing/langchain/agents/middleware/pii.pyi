from _typeshed import Incomplete
from collections.abc import Callable
from langchain.agents.middleware._redaction import PIIDetectionError as PIIDetectionError, PIIMatch as PIIMatch, detect_credit_card as detect_credit_card, detect_email as detect_email, detect_ip as detect_ip, detect_mac_address as detect_mac_address, detect_url as detect_url
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langgraph.runtime import Runtime
from typing import Any, Literal
from typing_extensions import override

__all__ = ['PIIDetectionError', 'PIIMatch', 'PIIMiddleware', 'detect_credit_card', 'detect_email', 'detect_ip', 'detect_mac_address', 'detect_url']

class PIIMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    apply_to_input: Incomplete
    apply_to_output: Incomplete
    apply_to_tool_results: Incomplete
    pii_type: Incomplete
    strategy: Incomplete
    detector: Incomplete
    def __init__(self, pii_type: Literal['email', 'credit_card', 'ip', 'mac_address', 'url'] | str, *, strategy: Literal['block', 'redact', 'mask', 'hash'] = 'redact', detector: Callable[[str], list[PIIMatch]] | str | None = None, apply_to_input: bool = True, apply_to_output: bool = False, apply_to_tool_results: bool = False) -> None: ...
    @property
    def name(self) -> str: ...
    @override
    def before_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def abefore_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def aafter_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
