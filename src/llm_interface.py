import json
import ollama

from pathlib import Path

from .models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class LLMInterface:
    def __init__(self, llm: str = "qwen3:0.6b") -> None:
        self.llm = llm

        if llm not in [m.model for m in ollama.list().models]:
            print(f"Installing model '{llm}'...")
            ollama.pull(llm)

    def load_dataset(self, dataset_path: str) -> StudentSearchResults:
        dataset_file = Path(dataset_path)
        dataset_json = json.loads(dataset_file.read_text())

        return StudentSearchResults(**dataset_json)

    def retrieve_source(self, file_path: str, first_character_index: int,
                        last_character_index: int) -> str:
        file = Path(file_path)
        return file.read_text()[first_character_index:last_character_index]

    def answer(self, query: str, context: str,
               print_result: bool = False) -> str:
        response = ollama.chat(
            model=self.llm,
            messages=[
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )

        if print_result:
            print(response.message.content)

        return response.message.content

    def answer_dataset(self, dataset: StudentSearchResults) \
            -> StudentSearchResultsAndAnswer:
        answers = []
        for entry in dataset.search_results:
            query = entry.question

            sources = [
                self.retrieve_source(
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index)
                for source in entry.retrieved_sources
            ]

            context = "\n\n===== SOURCE SEPARATOR =====\n\n".join(sources)

            answer = self.answer(query, context)

            answers.append(MinimalAnswer(**entry.model_dump(), answer=answer))

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )

        return output

    def save_answers(self, answers: StudentSearchResults,
                     save_directory: str) -> None:
        save_file = Path(save_directory)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        results_json = answers.model_dump()
        save_file.write_text(json.dumps(results_json, indent=4))
