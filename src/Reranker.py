from concurrent.futures import ThreadPoolExecutor
from flashrank import Ranker, RerankRequest
from typing import List

from .constants import (
    RERANKER_CACHE_DIR,
    RERANKER_CROP
)
from .IOUtils import IOUtils
from .models import MinimalSource


class Reranker:
    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2"
    ) -> None:
        self.ranker = Ranker(
            model_name=model_name,
            cache_dir=RERANKER_CACHE_DIR,
            log_level="ERROR"
        )

    def _prepare_and_run(self, query: str, sources: List[MinimalSource]) -> List[dict]:
        if not sources:
            return []

        def load_passage(item):
            idx, src = item
            src_data = src.model_dump()
            return {
                "id": idx,
                "text": IOUtils.get_text_from_file(**src_data)[:RERANKER_CROP],
                **src_data
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            passages = list(executor.map(load_passage, enumerate(sources)))

        rerank_request = RerankRequest(query=query, passages=passages)
        return self.ranker.rerank(rerank_request)

    def rerank_sources(
        self,
        query: str,
        sources: List[MinimalSource],
        min_score: float = -1
    ) -> List[MinimalSource]:
        reranked_results = self._prepare_and_run(query, sources)

        return [
            MinimalSource(
                file_path=result["file_path"],
                first_character_index=result["first_character_index"],
                last_character_index=result["last_character_index"]
            )
            for result in reranked_results if result["score"] >= min_score
        ]
