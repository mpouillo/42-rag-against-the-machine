#!/usr/bin/env python3

import asyncio
import fire
import sys

from pathlib import Path

from .indexer import Indexer
from .searcher import Searcher
from .llm_interface import LLMInterface
from .models import StudentSearchResults


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
        result = searcher.search(query, k, print_result=True)
        output = StudentSearchResults(search_results=[result], k=k)
        searcher.save_search_results(output, "data/output/search_result.json")

    def search_dataset(self, dataset_path: str,
                       save_directory: str, k: int = 10) -> None:
        searcher = Searcher(self.index_path)
        results = searcher.search_dataset(dataset_path, k)

        searcher.save_search_results(results, save_directory)
        print(f"Saved student_search_results to {save_directory}")

    def answer(self, query: str, path_to_context: str) -> None:
        llm = LLMInterface()
        context = Path(path_to_context)
        llm.answer(query, context.read_text(), print_result=True)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        llm = LLMInterface("llama3.2:1b")
        dataset = llm.load_dataset(student_search_results_path)
        answers = asyncio.run(llm.answer_dataset(dataset))
        llm.save_answers(answers, save_directory)

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
