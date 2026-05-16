import json
import ollama

from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

from .constants import SYSTEM_PROMPT
from .models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class LLMInterface:
    def __init__(self, llm: str = "qwen3:0.6b") -> None:
        self.llm = llm
        self.client = ollama.AsyncClient()
        self._file_cache = {}

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

    async def answer(self, query: str, context: str,
                     print_result: bool = False) -> str:
        response = ollama.chat(
            model=self.llm,
            messages=[{"role": "context", "content": context},
                      {"role": "user", "content": query}]
        )

        if print_result:
            print(response.message.content)

        return response.message.content

    async def answer_dataset(self, dataset: StudentSearchResults) \
            -> StudentSearchResultsAndAnswer:

        async def process_entry(entry):
            sources = [
                self.retrieve_source(
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index)
                for source in entry.retrieved_sources
            ]

            context = "\n\n\n".join(sources)
            response = await self.client.chat(
                model=self.llm,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Reference Text:\n{context}"
                     f"\n\nQuestion: {entry.question}"}
                ],
                options={
                    "keep_alive": -1,
                    "num_ctx": 512,
                    "temperature": 0.1,
                }
            )

            return MinimalAnswer(
                **entry.model_dump(), answer=response.message.content
            )

        tasks = [process_entry(entry) for entry in dataset.search_results]
        answers = await tqdm_asyncio.gather(*tasks,
                                            desc="Processing queries...")
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )

    def save_answers(self, answers: StudentSearchResults,
                     save_directory: str) -> None:
        save_file = Path(save_directory)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        results_json = answers.model_dump()
        save_file.write_text(json.dumps(results_json, indent=4))
