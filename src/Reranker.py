
from flashrank import Ranker, RerankRequest
from typing import List

from .constants import (
    RERANKER_THRESHOLD,
    RERANKER_CACHE_DIR,
    RERANKER_LLM_MODEL
)
from .IOUtils import IOUtils
from .models import MinimalSource


class Reranker:
    def __init__(
        self
    ) -> None:
        self.ranker = Ranker(model_name=RERANKER_LLM_MODEL,
                             cache_dir=RERANKER_CACHE_DIR,
                             log_level="ERROR")

    def filter_sources(
        self,
        query: str,
        sources: List[MinimalSource]
    ) -> List[str]:
        passages = [
            {"id": idx, "text": IOUtils.get_text_from_file(**src)}
            for idx, src in enumerate(sources)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)

        return [
            sources[doc["id"]] for doc in reranked_results
            if doc["score"] >= RERANKER_THRESHOLD
        ]

    def rerank_sources(
        self,
        query: str,
        sources: List[MinimalSource]
    ) -> List[MinimalSource]:
        passages = [
            {
                "id": idx,
                "text": IOUtils.get_text_from_file(**src.model_dump()),
                **src.model_dump()
            }
            for idx, src in enumerate(sources)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)

        return [
            MinimalSource(
                file_path=result["file_path"],
                first_character_index=result["first_character_index"],
                last_character_index=result["last_character_index"]
            )
            for result in reranked_results
        ]
