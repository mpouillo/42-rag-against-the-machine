import tempfile
import weakref
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from langchain.agents.middleware._execution import BaseExecutionPolicy, CodexSandboxExecutionPolicy as CodexSandboxExecutionPolicy, DockerExecutionPolicy as DockerExecutionPolicy, HostExecutionPolicy as HostExecutionPolicy
from langchain.agents.middleware._redaction import RedactionRule as RedactionRule
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, PrivateStateAttr, ResponseT
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.runtime import Runtime
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated, Any
from typing_extensions import NotRequired, override

__all__ = ['CodexSandboxExecutionPolicy', 'DockerExecutionPolicy', 'HostExecutionPolicy', 'RedactionRule', 'ShellToolMiddleware']

@dataclass
class _SessionResources:
    session: ShellSession
    tempdir: tempfile.TemporaryDirectory[str] | None
    policy: BaseExecutionPolicy
    finalizer: weakref.finalize = field(init=False, repr=False)
    def __post_init__(self) -> None: ...

class ShellToolState(AgentState[ResponseT]):
    shell_session_resources: NotRequired[Annotated[_SessionResources | None, UntrackedValue, PrivateStateAttr]]

@dataclass(frozen=True)
class CommandExecutionResult:
    output: str
    exit_code: int | None
    timed_out: bool
    truncated_by_lines: bool
    truncated_by_bytes: bool
    total_lines: int
    total_bytes: int

class ShellSession:
    def __init__(self, workspace: Path, policy: BaseExecutionPolicy, command: tuple[str, ...], environment: Mapping[str, str]) -> None: ...
    def start(self) -> None: ...
    def restart(self) -> None: ...
    def stop(self, timeout: float) -> None: ...
    def execute(self, command: str, *, timeout: float) -> CommandExecutionResult: ...

class _ShellToolInput(BaseModel):
    command: str | None
    restart: bool | None
    runtime: Annotated[Any, None]
    def validate_payload(self) -> _ShellToolInput: ...

class ShellToolMiddleware(AgentMiddleware[ShellToolState[ResponseT], ContextT, ResponseT]):
    state_schema = ShellToolState
    tools: Incomplete
    def __init__(self, workspace_root: str | Path | None = None, *, startup_commands: tuple[str, ...] | list[str] | str | None = None, shutdown_commands: tuple[str, ...] | list[str] | str | None = None, execution_policy: BaseExecutionPolicy | None = None, redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None, tool_description: str | None = None, tool_name: str = ..., shell_command: Sequence[str] | str | None = None, env: Mapping[str, Any] | None = None) -> None: ...
    @override
    def before_agent(self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    async def abefore_agent(self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    @override
    def after_agent(self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]) -> None: ...
    async def aafter_agent(self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]) -> None: ...
