from collections import defaultdict
from tqdm import tqdm
from typing import Any, Dict, List

from .BM25Index import BM25Index
from .constants import (
    MODEL_RERANKER,
    MODEL_REWRITER,
    MODEL_VECTOR
)
from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults
)
from .QueryRewriter import QueryRewriter
from .Reranker import Reranker
from .VectorIndex import VectorIndex


class RAGSearch:
    """Core RAG class used to search indices for relevant sources."""
    def __init__(
        self,
        index_directory: str,
        hybrid_retrieval: bool = False
    ) -> None:
        """
        Initialize indices and query rewriter.

        Args:
            index_directory (str): Path to index directory
            hybrid_retrieval (bool, default = False): Whether to use
            hybrid retrieval for search.

        Returns:
            None: None
        """
        self.hybrid_retrieval = hybrid_retrieval
        self._query_cache: Dict[str, MinimalSearchResults] = {}

        self.bm25 = BM25Index()
        self.reranker = Reranker(MODEL_RERANKER)
        self.rewriter = QueryRewriter(MODEL_REWRITER)

        self.bm25.load(index_directory)

        if self.hybrid_retrieval:
            self.vector = VectorIndex(MODEL_VECTOR)
            self.vector.load(index_directory)

    def compute_rrf(
        self,
        source_lists: List[List[MinimalSource]],
        rrf_k: float = 60.0
    ) -> List[MinimalSource]:
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
                    src.file_path,
                    src.first_character_index,
                    src.last_character_index
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
        pool = max(k * 10, 50)
        rerank_k = k * 2

        search_results = []
        for entry in tqdm(dataset.rag_questions, desc="Processing queries"):
            if entry.question in self._query_cache:
                search_results.append(self._query_cache[entry.question])
                continue

            query = self.rewriter.rewrite_query(entry.question)
            bm25_srcs = self.bm25.search(query, pool)

            if self.hybrid_retrieval:
                vector_srcs = self.vector.search(query, pool)
                retrieved_srcs = self.compute_rrf([bm25_srcs, vector_srcs])
            else:
                retrieved_srcs = bm25_srcs

            reranked_srcs = self.reranker.rerank_sources(
                entry.question, retrieved_srcs[:rerank_k]
            )

            src_result = MinimalSearchResults(
                question=entry.question,
                question_id=entry.question_id,
                retrieved_sources=reranked_srcs[:k]
            )

            search_results.append(src_result)
            self._query_cache[entry.question] = src_result

        return StudentSearchResults(search_results=search_results, k=k)
