"""Provides a command-line interface integrating the core RAG functions."""

import asyncio
import json

from .constants import (
    INDEX_DIRECTORY,
    INGEST_DIRECTORY,
    MODEL_ANSWER,
    SEARCH_DIRECTORY
)
from .IOUtils import IOUtils
from .models import (
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion
)
from .RAGAnswer import RAGAnswer
from .RAGEvaluate import RAGEvaluate
from .RAGIndex import RAGIndex
from .RAGSearch import RAGSearch


class RAGInterface(object):
    """Core RAG class used to provide main system interface functions."""

    def index(
        self,
        max_chunk_size: int = 2000
    ) -> None:
        """Index data from raw ingestion path and save it to disk.

        Args:
            max_chunk_size (int): Maximum size of output chunks.

        Raises:
            ValueError: If `max_chunk_size` is not a positive non-zero integer.
        """
        try:
            max_chunk_size = int(max_chunk_size)
            if max_chunk_size < 1:
                raise ValueError
        except ValueError:
            raise ValueError(
                "'max_chunk_size' must be a positive non-zero integer"
            )

        indexer = RAGIndex(INDEX_DIRECTORY)
        indexer.ingest_and_index(max_chunk_size, INGEST_DIRECTORY)
        print(f"Ingestion complete! Indices saved under {INDEX_DIRECTORY}")

    def search(
        self,
        query: str,
        k: int = 10
    ) -> str:
        """Search the index and retrieve top k matching sources.

        Args:
            query (str): The text to match against the local index.
            k (int): The maximum number of sources to provide. Defaults to 10.

        Returns:
            str: JSON-formatted string of the retrieved StudentSearchResults.

        Raises:
            ValueError: If `k` is not a positive non-zero integer.
        """
        query = str(query)

        try:
            k = int(k)
            if k < 1:
                raise ValueError
        except ValueError:
            raise ValueError("'k' must be a positive non-zero integer")

        searcher = RAGSearch(INDEX_DIRECTORY)
        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(question=query)
            ]
        )
        results = asyncio.run(searcher.search_dataset(dataset, k))
        return results.model_dump_json(indent=4)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = SEARCH_DIRECTORY
    ) -> None:
        """Search index and retrieve top k sources for each query in a dataset.

        Args:
            dataset_path (str): Path to the dataset to process.
            k (int): The number of sources to retrieve. Defaults to 10.
            save_directory (str): Directory to save search results.

        Raises:
            ValueError: If `k` is not a positive non-zero integer.
        """
        dataset_path = str(dataset_path)
        save_directory = str(save_directory)

        try:
            k = int(k)
            if k < 1:
                raise ValueError
        except ValueError:
            raise ValueError("'k' must be a positive non-zero integer")

        searcher = RAGSearch(INDEX_DIRECTORY)
        dataset = IOUtils.load_json_as_model(dataset_path, RagDataset)
        results = asyncio.run(searcher.search_dataset(dataset, k))

        filename = dataset_path.split("/")[-1]
        save_path = f"{save_directory}/{filename}"
        IOUtils.save_object_as_json(save_path, results)
        print(f"Saved {filename.split('.')[0]} to {save_path}")

    def answer(
        self,
        query: str,
        k: int = 10
    ) -> str:
        """Search index for a query, retrieve k sources, and answer via LLM.

        Args:
            query (str): The text to match against the local index.
            k (int): The number of sources to use for context. Defaults to 10.

        Returns:
            str: JSON-formatted string of the
                generated StudentSearchResultsAndAnswer.

        Raises:
            ValueError: If `k` is not a positive non-zero integer.
        """
        query = str(query)

        try:
            k = int(k)
            if k < 1:
                raise ValueError
        except ValueError:
            raise ValueError("'k' must be a positive non-zero integer")

        results = StudentSearchResults(**json.loads(self.search(query, k)))
        answers = asyncio.run(RAGAnswer(MODEL_ANSWER).answer_dataset(results))
        return answers.model_dump_json(indent=4)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = SEARCH_DIRECTORY
    ) -> None:
        """Answer dataset queries based on retrieved sources using an LLM.

        Args:
            student_search_results_path (str): Path to the processed dataset.
            save_directory (str): Directory to save final results.
        """
        student_search_results_path = str(student_search_results_path)
        save_directory = str(save_directory)

        dataset = IOUtils.load_json_as_model(
            student_search_results_path, StudentSearchResults
        )
        total = len(dataset.search_results)
        print(f"Loaded {total} questions from {student_search_results_path}")

        answers = asyncio.run(RAGAnswer(MODEL_ANSWER).answer_dataset(dataset))
        count = len(answers.search_results)
        print(f"Processed {count} of {total} questions")

        save_path = (f"{save_directory}/"
                     f"{student_search_results_path.split('/')[-1]}")
        IOUtils.save_object_as_json(save_path, answers)
        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int,
        max_context_length: int
    ) -> None:
        """Evaluate student search results based on a reference dataset.

        Args:
            student_search_results_path (str): Path to the predictions dataset.
            dataset_path (str): Path to the reference truth dataset.
            k (int): The limit of retrieved sources to factor into evaluation.
            max_context_length (int): Maximum length constraints to consider.

        Raises:
            ValueError: If `k` or `max_context_length`
                are not positive integers.
        """
        student_search_results_path = str(student_search_results_path)
        dataset_path = str(dataset_path)

        try:
            k = int(k)
            if k < 1:
                raise ValueError
        except ValueError:
            raise ValueError("'k' must be a positive non-zero integer")

        try:
            max_context_length = int(max_context_length)
            if max_context_length < 1:
                raise ValueError
        except ValueError:
            raise ValueError(
                "'max_context_length' must be a positive non-zero integer"
            )

        evaluator = RAGEvaluate(student_search_results_path, dataset_path)
        evaluator.validate(k, max_context_length)
        print()
        evaluator.evaluate()
