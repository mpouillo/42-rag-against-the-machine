#!/usr/bin/env python3

import asyncio
import fire
import json
import sys

from .constants import (
    CHUNK_FILEPATH,
    INDEX_PATH,
    PATH_TO_INGEST,
    SEARCH_SAVE_FILEPATH
)
from .indexer import Indexer
from .searcher import Searcher
from .llm_interface import Answerer


class RagInterface(object):
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(PATH_TO_INGEST)
        chunks = (indexer.chunkify("markdown", max_chunk_size)
                  + indexer.chunkify("python", max_chunk_size))
        indexer.save_chunks(chunks, CHUNK_FILEPATH)
        indexer.create_index(chunks, INDEX_PATH)
        print(f"Ingestion complete! Indices saved under {INDEX_PATH}")

    def search(self, query: str, k: int = 10) -> str:
        searcher = Searcher(INDEX_PATH)
        results = searcher.search(query, k)
        print(json.dumps(results.model_dump(), indent=4))
        return results.model_dump_json()

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = SEARCH_SAVE_FILEPATH) -> None:
        searcher = Searcher(INDEX_PATH)
        results = searcher.search_dataset(dataset_path, k)
        searcher.save_search_results(results, save_directory)
        print(f"Saved student_search_results to {save_directory}")

    def answer(self, query: str, k: int = 10) -> str:
        results = Searcher(INDEX_PATH).search(query, k)
        answers = Answerer().answer_dataset(results)
        print(json.dumps(answers.model_dump(), indent=4))
        return answers.model_dump_json()

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        answerer = Answerer()
        dataset = answerer.load_dataset(student_search_results_path)
        answers = asyncio.run(answerer.answer_dataset(dataset))
        answerer.save_answers(answers, save_directory)
        print(f"Saved student_search_results_and_answer to {save_directory}")

    def evaluate(self, student_answer_path: str, dataset_path: str,
                 k: int, max_context_length: int) -> None:
        pass


def main() -> None:
    fire.Fire(RagInterface)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        sys.exit(f"Error: {e}")
