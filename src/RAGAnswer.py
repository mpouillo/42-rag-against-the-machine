import ollama

from tqdm.asyncio import tqdm_asyncio

from .constants import (
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_FAILURE_ANSWER,
    LLM_SYSTEM_PROMPT
)
from .IOUtils import IOUtils
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from .Reranker import Reranker


class Answerer:
    def __init__(self, llm: str = "qwen3:0.6b") -> None:
        self.llm: str = llm
        self.client = ollama.AsyncClient()
        self.reranker = Reranker()

        if llm not in [m.model for m in ollama.list().models]:
            print(f"Installing model '{llm}'...")
            ollama.pull(llm)

    async def answer_dataset(self, dataset: StudentSearchResults) \
            -> StudentSearchResultsAndAnswer:

        async def process_entry(entry: MinimalSearchResults) -> MinimalAnswer:
            sources = [
                IOUtils.get_text_from_file(
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index
                )
                for source in entry.retrieved_sources
            ]

            context = self.reranker.filter_sources(entry.question, sources)[:3]
            if not context:
                return MinimalAnswer(**entry.model_dump(),
                                     answer=LLM_FAILURE_ANSWER)

            user_content = (
                "/no_think "
                f"# Instructions:\n{LLM_SYSTEM_PROMPT}\n\n\n"
                f"# Context:\n{"\n\n".join(context[::-1])}\n\n\n"
                f"# Question:\n{entry.question}\n\n\n"
            )
            response = await self.client.chat(
                model=self.llm,
                think=False,
                keep_alive=-1,
                messages=[{"role": "user", "content": user_content}],
                options={"temperature": LLM_TEMPERATURE,
                         "num_predict": LLM_NUM_PREDICT}
            )

            return MinimalAnswer(
                **entry.model_dump(), answer=response.message.content
            )

        tasks = [process_entry(entry) for entry in dataset.search_results]
        answers = await tqdm_asyncio.gather(*tasks, desc="Processing...")
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )
