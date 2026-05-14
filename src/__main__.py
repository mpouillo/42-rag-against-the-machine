#!/usr/bin/env python3

import fire
import sys

from .indexer import Indexer
from .searcher import Searcher
from .llm_interface import LLMInterface


class RagInterface(object):
    def __init__(self):
        self.index_path = "data/processed/bm25_index"
        self.chunk_filepath = "data/processed/chunks/chunks.json"
        self.path_to_process = "data/raw"

    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(self.path_to_process)
        chunks = (
            indexer.chunkify("markdown", max_chunk_size)
            + indexer.chunkify("python", max_chunk_size)
        )
        indexer.save_chunks(chunks, self.chunk_filepath)

        indexer.create_index(chunks, self.index_path)
        print(f"Ingestion complete! Indices saved under {self.index_path}")

    def search(self, query: str, k: int = 10) -> None:
        searcher = Searcher(self.index_path)
        searcher.search(query, k, print_result=True)

    def search_dataset(self, dataset_path: str,
                       save_directory: str, k: int = 10) -> None:
        searcher = Searcher(self.index_path)
        results = searcher.search_dataset(dataset_path, k)

        searcher.save_search_results(results, save_directory)
        print(f"Saved student_search_results to {save_directory}")

    def answer(self, query: str) -> None:
        llm = LLMInterface()
        print(llm.answer(query))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        llm = LLMInterface()
        dataset = llm.load_dataset()
        print(f"Loaded {len(dataset)} questions "
              f"from {student_search_results_path}")

        answers = llm.answer_dataset()
        print(f"Processed {len(answers)} of {len(dataset)} questions")

        print(f"Saved student_search_results_and_answer to {save_directory}")

    def evaluate(self) -> None:
        pass


def main() -> None:
    fire.Fire(RagInterface)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        sys.exit(f"Error: {e}")
