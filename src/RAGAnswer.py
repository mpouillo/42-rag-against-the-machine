import asyncio

from tqdm.asyncio import tqdm_asyncio

from .constants import (
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_FAILURE_ANSWER,
    LLM_SYSTEM_PROMPT,
    CONTEXT_TRIM
)
from .IOUtils import IOUtils
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from .LLMInterface import LLMInterface


class RAGAnswer(LLMInterface):
    def __init__(self, llm: str = "qwen3:0.6b") -> None:
        super().__init__(llm)

    async def answer_dataset(self, dataset: StudentSearchResults) \
            -> StudentSearchResultsAndAnswer:
        await self._check_ready()
        sem = asyncio.Semaphore(8)

        async def process_entry(entry: MinimalSearchResults) -> MinimalAnswer:
            async with sem:
                context = [
                    IOUtils.get_text_from_file(**src.model_dump())
                    for src in entry.retrieved_sources
                ][:CONTEXT_TRIM]

                if not context:
                    return MinimalAnswer(**entry.model_dump(),
                                        answer=LLM_FAILURE_ANSWER)

                user_content = (
                    "/no_think\n"
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
        answers = await tqdm_asyncio.gather(*tasks, desc="Answering queries")
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )
