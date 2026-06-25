"""Module used to query the LLM to answer questions using passed contexts."""

import asyncio

from tqdm.asyncio import tqdm_asyncio

from .constants import (
    LLM_CONTEXT_TRIM,
    LLM_FAILURE_ANSWER,
    LLM_NUM_PREDICT,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE
)
from .IOUtils import IOUtils
from .LLMInterface import LLMInterface
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class RAGAnswer(LLMInterface):
    """Core RAG class used to answer queries based on provided sources.

    Inherits from LLMInterface to interact with the chosen language model.
    """

    def __init__(
        self,
        model: str = "qwen3:0.6b"
    ) -> None:
        """Initialize the RAG Answerer and its underlying LLM client.

        Args:
            model (str): Name of the model to use. Defaults to "qwen3:0.6b".
        """
        super().__init__(model)

    async def answer_dataset(
        self,
        dataset: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Answer queries from a StudentSearchResults object using an LLM.

        Args:
            dataset (StudentSearchResults): The dataset of searches to process.

        Returns:
            StudentSearchResultsAndAnswer: Dataset containing LLM responses.
        """
        await self._check_ready()
        sem = asyncio.Semaphore(8)

        async def process_entry(entry: MinimalSearchResults) -> MinimalAnswer:
            """Process a single query to generate a context-augmented answer.

            Args:
                entry (MinimalSearchResults): A single minimal search result
                    containing the query and sources.

            Returns:
                MinimalAnswer: The result containing the generated LLM text.
            """
            async with sem:
                context = [
                    IOUtils.get_text_from_file(**src.model_dump())
                    for src in entry.retrieved_sources
                ][:LLM_CONTEXT_TRIM]

                if not context:
                    return MinimalAnswer(**entry.model_dump(),
                                         answer=LLM_FAILURE_ANSWER)

                full_context = "\n\n".join(context[::-1])
                user_content = (
                    "/no_think\n"
                    f"# Instructions:\n{LLM_SYSTEM_PROMPT}\n\n\n"
                    f"# Context:\n{full_context}\n\n\n"
                    f"# Question:\n{entry.question}\n\n\n"
                )

                response = await self.client.chat(
                    model=self.model,
                    think=False,
                    keep_alive=-1,
                    messages=[{"role": "user", "content": user_content}],
                    options={"temperature": LLM_TEMPERATURE,
                             "num_predict": LLM_NUM_PREDICT}
                )

                return MinimalAnswer(
                    **entry.model_dump(), answer=str(response.message.content)
                )

        tasks = [process_entry(entry) for entry in dataset.search_results]
        answers = await tqdm_asyncio.gather(*tasks, desc="Answering queries")

        return StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )
