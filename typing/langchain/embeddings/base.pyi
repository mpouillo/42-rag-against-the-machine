from langchain_core.embeddings import Embeddings as Embeddings
from typing import Any

__all__ = ['Embeddings', 'init_embeddings']

def init_embeddings(model: str, *, provider: str | None = None, **kwargs: Any) -> Embeddings: ...
