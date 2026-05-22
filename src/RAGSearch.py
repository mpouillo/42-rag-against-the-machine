from tqdm import tqdm
from collections import defaultdict
from typing import List
from src import IOUtils

from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults
)
from .BM25Index import BM25Index
from .VectorIndex import VectorIndex
from .Reranker import Reranker
from .QueryRewriter import QueryRewriter
from .IOUtils import IOUtils
from .constants import RERANKER_LLM_MODEL


class RAGSearch:
    def __init__(
        self,
        index_directory: str,
        hybrid_retrieval: bool = False
    ):
        self.hybrid_retrieval = hybrid_retrieval
        self._query_cache = {}

        self.bm25 = BM25Index()
        self.reranker = Reranker(RERANKER_LLM_MODEL)
        self.rewriter = QueryRewriter()

        self.bm25.load(index_directory)

        if self.hybrid_retrieval:
            self.vector = VectorIndex()
            self.vector.load(index_directory)

    def compute_rrf(
        self,
        source_lists: List[List[MinimalSource]],
        rrf_k: int = 60
    ) -> List[MinimalSource]:
        rrf_scores = defaultdict(float)
        source_map = {}

        for source_list in source_lists:
            for rank, src in enumerate(source_list, start=1):
                footprint = (src.file_path, src.first_character_index, src.last_character_index)
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
