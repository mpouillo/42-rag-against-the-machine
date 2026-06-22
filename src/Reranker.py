from flashrank import Ranker, RerankRequest
from typing import Any, Dict, List, cast

from .constants import RERANKER_CACHE_DIR


class Reranker:
    """Source reranking pipeline to improve search engine retrieval."""
    def __init__(
        self,
        model: str = "ms-marco-MiniLM-L-12-v2"
    ) -> None:
        """
        Initialize ranker model.

        Args:
            model (str): Name of the LLM model to use

        Returns:
            None: None
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
        Rerank sources based on LLM scoring.

        Args:
            query (str): The text to match against sources
            sources (List[MinimalSource]): List of sources to rerank

        Returns:
            List[MinimalSource]: List of reranked sources
        """

        for idx, source in enumerate(sources):
            source["id"] = idx
        rerank_request = RerankRequest(query=query, passages=sources)
        reranked_results = self.ranker.rerank(rerank_request)

        return cast(List[Dict[str, Any]], reranked_results)
