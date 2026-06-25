"""Reranker component for prioritizing query search results."""

from flashrank import Ranker, RerankRequest
from typing import Any, Dict, List, cast

from .constants import RERANKER_CACHE_DIR


class Reranker:
    """Source reranking pipeline to improve search engine retrieval quality.

    Attributes:
        ranker (Ranker): The instantiated FlashRank model used for scoring.
    """

    def __init__(
        self,
        model: str = "ms-marco-MiniLM-L-12-v2"
    ) -> None:
        """Initialize ranker model and establish the cache directory.

        Args:
            model (str): Name of the specific LLM ranking model to use.
        """
        self.ranker = Ranker(
            model_name=model,
            cache_dir=RERANKER_CACHE_DIR,
            log_level="ERROR"
        )

    def rerank_sources(
        self,
        query: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank an initial list of chunk sources
        based on cross-encoder LLM scoring.

        Args:
            query (str): The search text to match against retrieved sources.
            sources (List[Dict[str, Any]]):List of source dictionaries.

        Returns:
            List[Dict[str, Any]]: Re-ordered list of reranked sources.
        """
        for idx, source in enumerate(sources):
            source["id"] = idx
        rerank_request = RerankRequest(query=query, passages=sources)
        reranked_results = self.ranker.rerank(rerank_request)

        return cast(List[Dict[str, Any]], reranked_results)
