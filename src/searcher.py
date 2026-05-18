import bm25s
import json
import Stemmer
import sys

from pathlib import Path
from tqdm import tqdm

from .constants import BM25_DIRECTORY
from .models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion
)


class Searcher:
    def __init__(
        self,
        index_path: str
    ) -> None:
        try:
            bm25_path = Path(index_path) / BM25_DIRECTORY
            self.retriever = bm25s.BM25.load(str(bm25_path), load_corpus=True)
            self.stemmer = Stemmer.Stemmer("english")
        except Exception:
            sys.exit("No index or chunks found")

    def search(
        self,
        entry: UnansweredQuestion,
        k: int
    ) -> MinimalSearchResults:
        query_tokens = bm25s.tokenize(entry.question, stemmer=self.stemmer)
        results, _ = self.retriever.retrieve(query_tokens, k=k)

        retrieved_sources = [
            MinimalSource(**results[0, i]) for i in range(results.shape[1])
        ]

        return MinimalSearchResults(
            question_id=entry.question_id,
            question=entry.question,
            retrieved_sources=retrieved_sources
        )

    def search_dataset(
        self,
        dataset_path: str,
        k: int
    ) -> StudentSearchResults:
        dataset_file = Path(dataset_path)
        dataset_json = json.loads(dataset_file.read_text())
        dataset = RagDataset(**dataset_json)

        search_results = [
            self.search(entry, k)
            for entry in tqdm(dataset.rag_questions, desc="Searching...")
        ]

        return StudentSearchResults(search_results=search_results, k=k)
