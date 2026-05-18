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
    def __init__(self, index_path: str) -> None:
        bm25_path = Path(index_path) / BM25_DIRECTORY
        try:
            self.retriever = bm25s.BM25.load(str(bm25_path), load_corpus=True)
        except Exception:
            sys.exit("no index or chunks found")

    def search(self, entry: UnansweredQuestion,
               k: int) -> MinimalSearchResults:
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(entry.question, stemmer=stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        retrieved_sources = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i] # noqa
            retrieved_sources.append(MinimalSource(**doc))

        result = MinimalSearchResults(
            question_id=entry.question_id,
            question=entry.question,
            retrieved_sources=retrieved_sources
        )

        return result

    def search_dataset(self, dataset_path: str,
                       k: int) -> StudentSearchResults:
        dataset_file = Path(dataset_path)
        dataset_json = json.loads(dataset_file.read_text())
        dataset = RagDataset(**dataset_json)

        results = []
        for entry in tqdm(dataset.rag_questions, desc="Processing..."):
            results.append(self.search(entry, k))

        return StudentSearchResults(search_results=results, k=k)

    def save_search_results(self,
                            search_results: StudentSearchResults,
                            save_directory: str) -> None:
        save_file = Path(save_directory) / "dataset_docs_public.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        results_json = search_results.model_dump()
        save_file.write_text(json.dumps(results_json, indent=4))
