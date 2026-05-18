import json
import ollama

from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from typing import Dict

from .constants import LLM_NUM_PREDICT, LLM_TEMPERATURE, LLM_SYSTEM_PROMPT
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class Answerer:
    def __init__(self, llm: str = "qwen3:0.6b") -> None:
        self.llm: str = llm
        self.client = ollama.AsyncClient()
        self._file_cache: Dict[str, str] = {}

        if llm not in [m.model for m in ollama.list().models]:
            print(f"Installing model '{llm}'...")
            ollama.pull(llm)

    def load_dataset(self, dataset_path: str) -> StudentSearchResults:
        dataset_file = Path(dataset_path)
        dataset_json = json.loads(dataset_file.read_text())
        return StudentSearchResults(**dataset_json)

    def retrieve_source(self, file_path: str, first_character_index: int,
                        last_character_index: int) -> str:
        if file_path not in self._file_cache:
            self._file_cache[file_path] = Path(file_path).read_text()
        text = self._file_cache[file_path]
        return text[first_character_index:last_character_index]

    async def answer_dataset(self, dataset: StudentSearchResults) \
            -> StudentSearchResultsAndAnswer:

        async def process_entry(entry: MinimalSearchResults) -> MinimalAnswer:
            sources = [
                self.retrieve_source(
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index)
                for source in entry.retrieved_sources
            ]

            context = "\n\n\n".join(sources)
            user_content = (
                f"Instructions: {LLM_SYSTEM_PROMPT}\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {entry.question}\n\n"
                f"Remember: {LLM_SYSTEM_PROMPT}\n\n"
                "Answer the question now:"
            )
            response = await self.client.chat(
                model=self.llm,
                messages=[{"role": "user", "content": user_content}],
                options={"temperature": LLM_TEMPERATURE, "num_predict": LLM_NUM_PREDICT}
            )

            return MinimalAnswer(
                **entry.model_dump(), answer=response.message.content
            )

        tasks = [process_entry(entry) for entry in dataset.search_results]
        answers = await tqdm_asyncio.gather(*tasks, desc="Processing...")
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )

    def save_answers(self, answers: StudentSearchResults,
                     save_directory: str) -> None:
        save_file = Path(save_directory) / "dataset_docs_public.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        results_json = answers.model_dump()
        save_file.write_text(json.dumps(results_json, indent=4))
