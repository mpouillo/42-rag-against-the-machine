from _typeshed import Incomplete
from langchain.agents.middleware.types import AgentMiddleware as AgentMiddleware, AgentState as AgentState, ContextT as ContextT, ResponseT as ResponseT, StateT as StateT
from langchain_core.messages import ToolCall
from langgraph.runtime import Runtime as Runtime
from typing import Any, Literal, Protocol
from typing_extensions import NotRequired, TypedDict

class Action(TypedDict):
    name: str
    args: dict[str, Any]

class ActionRequest(TypedDict):
    name: str
    args: dict[str, Any]
    description: NotRequired[str]

DecisionType: Incomplete

class ReviewConfig(TypedDict):
    action_name: str
    allowed_decisions: list[DecisionType]
    args_schema: NotRequired[dict[str, Any]]

class HITLRequest(TypedDict):
    action_requests: list[ActionRequest]
    review_configs: list[ReviewConfig]

class ApproveDecision(TypedDict):
    type: Literal['approve']

class EditDecision(TypedDict):
    type: Literal['edit']
    edited_action: Action

class RejectDecision(TypedDict):
    type: Literal['reject']
    message: NotRequired[str]

class RespondDecision(TypedDict):
    type: Literal['respond']
    message: str
Decision = ApproveDecision | EditDecision | RejectDecision | RespondDecision

class HITLResponse(TypedDict):
    decisions: list[Decision]

class _DescriptionFactory(Protocol):
    def __call__(self, tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[ContextT]) -> str: ...

class InterruptOnConfig(TypedDict):
    allowed_decisions: list[DecisionType]
    description: NotRequired[str | _DescriptionFactory]
    args_schema: NotRequired[dict[str, Any]]

class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    interrupt_on: Incomplete
    description_prefix: Incomplete
    def __init__(self, interrupt_on: dict[str, bool | InterruptOnConfig], *, description_prefix: str = 'Tool execution requires approval') -> None: ...
    def after_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def aafter_model(self, state: AgentState[Any], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
