from tqdm import tqdm

from .models import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults
)
from .BM25Interface import BM25Interface
from .VectorSearcher import VectorSearcher
from .Reranker import Reranker


class RAGSearch:
    def __init__(
        self,
        index_directory: str
    ):
        self.bm25 = BM25Interface()
        self.bm25.load(index_directory)
        self.vector = VectorSearcher("all-MiniLM-L6-v2")
        self.vector.load(index_directory)
        self.reranker = Reranker()

    def search_dataset(
        self,
        dataset: RagDataset,
        k: int
    ) -> StudentSearchResults:

        pool = k * 3

        search_results = []
        for entry in tqdm(dataset.rag_questions):
            bm25_res = self.bm25.search(entry, pool)
            vector_res = self.vector.search(entry.question, pool)
            comb = list(set(bm25_res + vector_res))
            reranked = self.reranker.filter_sources(entry.question, comb)
            search_results.append(
                MinimalSearchResults(
                    question=entry.question,
                    question_id=entry.question_id,
                    retrieved_sources=reranked
                )
            )

        return StudentSearchResults(search_results=search_results, k=k)
