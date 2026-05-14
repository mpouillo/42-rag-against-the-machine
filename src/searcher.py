import bm25s
import json
import Stemmer
import sys

from pathlib import Path
from typing import List

from .models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion
)


class Searcher:
    def __init__(self, index_path: str) -> None:
        try:
            self.retriever = bm25s.BM25.load(index_path, load_corpus=True)
        except Exception:
            sys.exit("no index or chunks found")

    def search(self, query: str, k: int,
               print_result: bool = False) -> MinimalSearchResults:
        question = UnansweredQuestion(question=query)

        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        retrieved_sources = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i] # noqa
            retrieved_sources.append(MinimalSource(**doc))

        result = MinimalSearchResults(
            question_id=question.question_id,
            question=query,
            retrieved_sources=retrieved_sources
        )

        if print_result:
            print(json.dumps(result.model_dump(), indent=4))

        return result

    def search_dataset(self, dataset_path: str,
                       k: int) -> List[StudentSearchResults]:
        dataset_file = Path(dataset_path)
        dataset_json = json.loads(dataset_file.read_text())
        dataset = RagDataset(**dataset_json)
        results = [self.search(d.question, k) for d in dataset.rag_questions]

        return StudentSearchResults(search_results=results, k=k)

    def save_search_results(self,
                            search_results: StudentSearchResults,
                            save_directory: str) -> None:
        save_file = Path(save_directory)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        results_json = search_results.model_dump()
        save_file.write_text(json.dumps(results_json, indent=4))
