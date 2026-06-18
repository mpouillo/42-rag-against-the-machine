from collections import defaultdict
from tqdm import tqdm
from typing import Any, Dict, List

from src.constants import MODEL_RERANKER

from .Reranker import Reranker

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
        """
        Initialize indices and query rewriter.
        If environment variable "HYBRID_RETRIEVAL" is set to True,
        uses hybrid retrieval for search computation.

        Args:
            index_directory (str): Path to index directory

        Returns:
            None: None
        """
        self.bm25 = BM25Searcher(index_dir)
        self.chroma = ChromaSearcher(index_dir)
        self.reranker = Reranker(MODEL_RERANKER)
        self._cache: Dict[str, MinimalSearchResults] = {}

    def compute_rrf(
        self,
        source_lists: List[List[Dict[str, Any]]],
        rrf_k: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Compute RRF (Reciprocal Rank Fusion) scores for each list of
        MinimalSource objects and sort them.

        Args:
            source_lists (List[List[MinimalSource]]): Lists of MinimalSource
            objects to be computed.
            rrf_k (float, default=60.0): Weight constant

        Returns:
            List[MinimalSource]: Fused and sorted sources
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

    def search_dataset(
        self,
        dataset: RagDataset,
        k: int
    ) -> StudentSearchResults:
        """
        Search index and retrieve top k matching sources for each query
        in the provided dataset.

        Args:
            dataset (RagDataset): The dataset to process
            k (int): The number of sources to retrieve

        Returns:
            StudentSearchResults: Pydantic object containing the input
            dataset with retrieved sources appended
        """
        search_results = []

        for entry in tqdm(
            dataset.rag_questions,
            desc="Searching database..."
        ):
            query = entry.question
            bm25_results = self.bm25.search(query, 1000)
            chroma_results = self.chroma.search(query, 1000)
            rrf_results = self.compute_rrf([bm25_results, chroma_results])
            reranked_results = self.reranker.rerank_sources(query, rrf_results[:k * 2])
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

            search_results.append(result)
            self._cache[query] = result

        return StudentSearchResults(search_results=search_results, k=k)
