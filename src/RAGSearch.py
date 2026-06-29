"""Module connecting queries to the system's indices to find documents."""

from collections import defaultdict
import os
from tqdm.asyncio import tqdm
from typing import Any, Dict, List

from .constants import MODEL_RERANKER, MODEL_REWRITER
from .Reranker import Reranker
from .QueryRewriter import QueryRewriter
from .BM25Pipeline import BM25Pipeline
from .ChromaPipeline import ChromaPipeline

from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion
)


class RAGSearch:
    """Core RAG class used to search indices for relevant sources.

    Attributes:
        bm25 (BM25Pipeline): The lexical indexer for fetching chunks.
        chroma (ChromaPipeline): The semantic indexer for fetching chunks.
        reranker (Reranker): The model instance to score and re-sort documents.
        rewriter (QueryRewriter): Pipeline to expand query variations.
    """

    def __init__(
        self,
        index_dir: str,
    ) -> None:
        """Initialize search components pointing to the target index directory.

        Args:
            index_dir (str): Target directory containing pipeline index data.
        """
        self.bm25 = BM25Pipeline(index_dir)
        self.chroma = ChromaPipeline(index_dir)
        self.reranker = Reranker(MODEL_RERANKER)
        self.rewriter = QueryRewriter(MODEL_REWRITER)
        self._cache: Dict[str, MinimalSearchResults] = {}

    def compute_rrf(
        self,
        source_lists: List[List[Dict[str, Any]]],
        rrf_k: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Calculate the Reciprocal Rank Fusion (RRF) across multi-method outputs.

        Args:
            source_lists (List[List[Dict[str, Any]]]): Lists of search outputs.
            rrf_k (float): Smoothing constant applied during fusion scoring.

        Returns:
            List[Dict[str, Any]]: Combined and ranked search results.
        """
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

    async def search_single_entry(
        self,
        entry: AnsweredQuestion | UnansweredQuestion,
        k: int
    ) -> MinimalSearchResults:
        """
        Process an individual query using multi-method retrieval and reranking.

        Args:
            entry (AnsweredQuestion | UnansweredQuestion): The query object.
            k (int): Number of top sources to retain post-reranking.

        Returns:
            MinimalSearchResults: Packaged results referencing metadata bounds.
        """
        if entry.question in self._cache:
            return self._cache[entry.question]

        pool = max(100, k * 10)

        if os.environ.get("RAG_BONUS") in ["True", "1"]:
            expanded_queries = await self.rewriter.rewrite_query(
                entry.question
            )
            query = "\n".join(expanded_queries)
        else:
            query = entry.question

        bm25_results = self.bm25.search(query, pool)
        chroma_results = self.chroma.search(query, pool)
        rrf_results = self.compute_rrf([bm25_results, chroma_results])
        reranked_results = self.reranker.rerank_sources(
            entry.question, rrf_results[:pool // 2]
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
        """Execute search asynchronously across an entire dataset of queries.

        Args:
            dataset (RagDataset): The collection of user queries to batch.
            k (int): Standard limit of documents retrieved per query.

        Returns:
            StudentSearchResults: The resulting structure.
        """
        tasks = [
            self.search_single_entry(entry, k)
            for entry in dataset.rag_questions
        ]

        search_results = []

        try:
            for future in tqdm.as_completed(
                tasks,
                desc="Searching database..."
            ):
                result = await future
                search_results.append(result)

        finally:
            if hasattr(self.rewriter, "client"):
                client: Any = self.rewriter.client
                if hasattr(client, "close"):
                    await client.close()

        return StudentSearchResults(search_results=search_results, k=k)
