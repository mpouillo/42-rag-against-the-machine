import asyncio
import json

from src import RAGAnswer

from .constants import INDEX_DIRECTORY, INGEST_DIRECTORY, SEARCH_DIRECTORY
from .IOUtils import IOUtils
from .models import RagDataset, StudentSearchResults, UnansweredQuestion
from .RAGIndex import RAGIndex
from .RAGAnswer import RAGAnswer
from .RAGSearch import RAGSearch
from .RAGEvaluate import RAGEvaluate


class RAGInterface(object):
    def index(
        self,
        max_chunk_size: int = 2000
    ) -> None:
        indexer = RAGIndex(INDEX_DIRECTORY)
        indexer.index_and_save(max_chunk_size, INGEST_DIRECTORY)

    def search(
        self,
        query: str,
        k: int = 10
    ) -> str:
        searcher = RAGSearch(INDEX_DIRECTORY)
        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(question=query)
            ]
        )
        results = searcher.search_dataset(dataset, k)
        return results.model_dump_json(indent=4)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = SEARCH_DIRECTORY
    ) -> None:
        searcher = RAGSearch(INDEX_DIRECTORY)
        dataset = IOUtils.load_json_as_model(dataset_path, RagDataset)
        results = searcher.search_dataset(dataset, k)

        filename = dataset_path.split("/")[-1]
        save_path = f"{save_directory}/{filename}"
        IOUtils.save_object_as_json(save_path, results)
        print(f"Saved {filename.split(".")[0]} to {save_path}")

    def answer(
        self,
        query: str,
        k: int = 10
    ) -> str:
        results = StudentSearchResults(**json.loads(self.search(query, k)))
        answers = asyncio.run(RAGAnswer().answer_dataset(results))
        return answers.model_dump_json(indent=4)

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

        answers = asyncio.run(RAGAnswer().answer_dataset(dataset))
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
        evaluator = RAGEvaluate(student_answer_path, dataset_path)
        evaluator.validate(k, max_context_length)
        print()
        evaluator.evaluate()
