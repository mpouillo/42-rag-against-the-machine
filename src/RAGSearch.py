import asyncio
from collections import defaultdict
from tqdm.asyncio import tqdm  # Use the async-native tqdm wrapper
from typing import Any, Dict, List

from .constants import MODEL_RERANKER
from .Reranker import Reranker
from .QueryRewriter import QueryRewriter
from .BM25Search import BM25Searcher
from .ChromaSearch import ChromaSearcher

from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults
)


class RAGSearch:
    """Core RAG class used to search indices for relevant sources."""
    def __init__(
        self,
        index_dir: str,
    ) -> None:
        self.bm25 = BM25Searcher(index_dir)
        self.chroma = ChromaSearcher(index_dir)
        self.reranker = Reranker(MODEL_RERANKER)
        self.rewriter = QueryRewriter()
        self._cache: Dict[str, MinimalSearchResults] = {}

    def compute_rrf(
        self,
        source_lists: List[List[Dict[str, Any]]],
        rrf_k: float = 60.0
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[Any, float] = defaultdict(float)
        source_map = {}

        for source_list in source_lists:
            for rank, src in enumerate(source_list, start=1):
                footprint = (
                    src.get("file_path"),
                    src.get("first_character_index"),
                    src.get("last_character_index")
                )
                rrf_scores[footprint] += 1.0 / (rrf_k + rank)
                source_map[footprint] = src

        sorted_footprints = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )

        return [source_map[fp] for fp in sorted_footprints]

    async def search_single_entry(self, entry: Any, k: int) -> MinimalSearchResults:
        """Helper async task processing an individual query."""
        # Await the rewriter cleanly without restarting the loop
        expanded_list = await self.rewriter.rewrite_query(entry.question)
        query = "\n".join(expanded_list)

        # Keep the database lookups and reranking synchronous as they were
        bm25_results = self.bm25.search(query, 1000)
        chroma_results = self.chroma.search(query, 1000)
        rrf_results = self.compute_rrf([bm25_results, chroma_results])

        reranked_results = self.reranker.rerank_sources(
            entry.question, rrf_results[:k * 2]
        )

        minimal_sources = [
            MinimalSource(
                file_path=src.get("file_path", ""),
                first_character_index=src.get("first_character_index", 0),
                last_character_index=src.get("last_character_index", 0)
            )
            for src in reranked_results
        ][:k]

        if minimal_sources and len(minimal_sources) < k:
            padding = k - len(minimal_sources)
            minimal_sources.extend([minimal_sources[-1]] * padding)

        result = MinimalSearchResults(
            question=entry.question,
            question_id=entry.question_id,
            retrieved_sources=minimal_sources
        )

        self._cache[entry.question] = result
        return result

    async def search_dataset(
        self,
        dataset: RagDataset,
        k: int
    ) -> StudentSearchResults:
        """
        Search index asynchronously using tqdm to track progress.
        """
        tasks = [
            self.search_single_entry(entry, k)
            for entry in dataset.rag_questions
        ]

        search_results = []

        try:
            # tqdm.asyncio.as_completed yields results as they finish
            # across the single persistent background event loop
            for future in tqdm.as_completed(
                tasks,
                desc="Searching database (Async)..."
            ):
                result = await future
                search_results.append(result)

        finally:
            # CRITICAL: Gracefully shut down your QueryRewriter HTTP client
            # connection pool when the dataset processing completes.
            if hasattr(self.rewriter, "client") and hasattr(self.rewriter.client, "close"):
                await self.rewriter.client.close()

        return StudentSearchResults(search_results=search_results, k=k)