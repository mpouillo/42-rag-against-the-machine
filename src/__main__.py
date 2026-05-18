#!/usr/bin/env python3

import asyncio
import fire
import json
import sys

from .answerer import Answerer
from .constants import (
    CHUNK_PATH,
    INDEX_DIRECTORY,
    INGEST_DIRECTORY,
    SEARCH_DIRECTORY
)
from .evaluator import Evaluator
from .indexer import Indexer
from .models import UnansweredQuestion
from .searcher import Searcher


class RagInterface(object):
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(INGEST_DIRECTORY)
        chunks = (indexer.chunkify("markdown", max_chunk_size)
                  + indexer.chunkify("python", max_chunk_size))
        indexer.save_chunks(chunks, CHUNK_PATH)
        indexer.create_index(chunks, INDEX_DIRECTORY)
        print(f"Ingestion complete! Indices saved under {INDEX_DIRECTORY}")

    def search(self, query: str, k: int = 10) -> str:
        searcher = Searcher(INDEX_DIRECTORY)
        entry = UnansweredQuestion(question=query)
        results = searcher.search(entry, k)
        print(json.dumps(results.model_dump(), indent=4))
        return results.model_dump_json()

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = SEARCH_DIRECTORY) -> None:
        searcher = Searcher(INDEX_DIRECTORY)
        results = searcher.search_dataset(dataset_path, k)
        searcher.save_search_results(results, save_directory)
        print("Saved student_search_results to "
              f"{save_directory}/dataset_docs_public.json")

    def answer(self, query: str, k: int = 10) -> str:
        results = Searcher(INDEX_DIRECTORY).search(query, k)
        answers = Answerer().answer_dataset(results)
        print(json.dumps(answers.model_dump(), indent=4))
        return answers.model_dump_json()

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        answerer = Answerer()
        dataset = answerer.load_dataset(student_search_results_path)
        total = len(dataset.search_results)
        print(f"Loaded {total} questions from {student_search_results_path}")
        answers = asyncio.run(answerer.answer_dataset(dataset))
        count = len(answers.search_results)
        print(f"Processed {count} of {total} questions")
        answerer.save_answers(answers, save_directory)
        print("Saved student_search_results_and_answer to "
              f"{save_directory}/dataset_docs_public.json")

    def evaluate(self, student_answer_path: str, dataset_path: str,
                 k: int, max_context_length: int) -> None:
        evaluator = Evaluator(student_answer_path, dataset_path)
        evaluator.validate(max_context_length)
        print()
        evaluator.evaluate(k)


def main() -> None:
    fire.Fire(RagInterface)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        sys.exit(f"Error: {e}")
