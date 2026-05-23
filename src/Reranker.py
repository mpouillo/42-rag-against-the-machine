from concurrent.futures import ThreadPoolExecutor
from flashrank import Ranker, RerankRequest
from typing import Any, Dict, List, Tuple

from .constants import (
    RERANKER_CACHE_DIR,
    RERANKER_CROP
)
from .IOUtils import IOUtils
from .models import MinimalSource


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

    def _load_passages(
        self,
        sources: List[MinimalSource]
    ) -> List[Dict[str, Any]]:
        """
        Helper function to load corpus from sources.

        Args:
            sources (list[MinimalSource]): List of sources to load

        Returns:
            List[Dict[str, Any]]: Corpus of sources with added text data
        """
        if not sources:
            return []

        def load_passage(
            item: Tuple[int, MinimalSource]
        ) -> Dict[str, Any]:
            """
            Helper function to load source text (limited to RERANKER_CROP char)
            and return a dict of the total data.
            """
            idx, src = item
            src_data = src.model_dump()
            return {
                "id": idx,
                "text": IOUtils.get_text_from_file(**src_data)[:RERANKER_CROP],
                **src_data
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            passages = list(executor.map(load_passage, enumerate(sources)))

        return passages

    def rerank_sources(
        self,
        query: str,
        sources: List[MinimalSource],
        min_score: float = -1
    ) -> List[MinimalSource]:
        """
        Rerank sources based on LLM scoring.

        Args:
            query (str): The text to match against sources
            sources (List[MinimalSource]): List of sources to rerank
            min_score (float, default=-1): Minimum score to filter sources

        Returns:
            List[MinimalSource]: List of reranked sources
        """

        passages = self._load_passages(sources)
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)

        return [
            MinimalSource(
                file_path=result["file_path"],
                first_character_index=result["first_character_index"],
                last_character_index=result["last_character_index"]
            )
            for result in reranked_results if result["score"] >= min_score
        ]
