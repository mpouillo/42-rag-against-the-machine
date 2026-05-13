from _typeshed import Incomplete
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT

__all__ = ['FilesystemFileSearchMiddleware']

class FilesystemFileSearchMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    root_path: Incomplete
    use_ripgrep: Incomplete
    max_file_size_bytes: Incomplete
    glob_search: Incomplete
    grep_search: Incomplete
    tools: Incomplete
    def __init__(self, *, root_path: str, use_ripgrep: bool = True, max_file_size_mb: int = 10) -> None: ...
