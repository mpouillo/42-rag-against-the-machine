#!/usr/bin/env python3

import asyncio
import fire
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
from .ioutils import IOUtils
from .models import UnansweredQuestion, StudentSearchResults
from .searcher import Searcher


class RagInterface(object):
    def index(
        self,
        max_chunk_size: int = 2000
    ) -> None:
        chunks = (
            Indexer.chunkify(INGEST_DIRECTORY, "markdown", max_chunk_size)
            + Indexer.chunkify(INGEST_DIRECTORY, "python", max_chunk_size)
        )
        Indexer.save_chunks(CHUNK_PATH, chunks)
        Indexer.create_index(chunks, INDEX_DIRECTORY)
        print(f"Ingestion complete! Indices saved under {INDEX_DIRECTORY}")

    def search(
        self,
        query: str,
        k: int = 10
    ) -> str:
        searcher = Searcher(INDEX_DIRECTORY)
        entry = UnansweredQuestion(question=query)
        results = searcher.search(entry, k).model_dump_json(indent=4)
        print(results)
        return results

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = SEARCH_DIRECTORY
    ) -> None:
        searcher = Searcher(INDEX_DIRECTORY)
        results = searcher.search_dataset(dataset_path, k)

        save_path = f"{save_directory}/{dataset_path.split("/")[-1]}"
        IOUtils.save_object_as_json(save_path, results)
        print(f"Saved student_search_results to {save_path}")

    def answer(
        self,
        query: str,
        k: int = 10
    ) -> str:
        searcher = Searcher(INDEX_DIRECTORY)
        results = searcher.search(query, k)
        answers = Answerer().answer_dataset(results).model_dump_json(indent=4)
        print(answers)
        return answers

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        dataset = IOUtils.load_json_as_model(
            student_search_results_path, StudentSearchResults
        )
        total = len(dataset.search_results)
        print(f"Loaded {total} questions from {student_search_results_path}")

        answerer = Answerer()
        answers = asyncio.run(answerer.answer_dataset(dataset))
        count = len(answers.search_results)
        print(f"Processed {count} of {total} questions")

        save_path = (f"{save_directory}/"
                     f"{student_search_results_path.split("/")[-1]}")
        IOUtils.save_object_as_json(save_path, answers)
        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int,
        max_context_length: int
    ) -> None:
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
