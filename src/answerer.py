import json
import ollama

from flashrank import Ranker, RerankRequest
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, List

from .constants import (
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_FAILURE_ANSWER,
    LLM_SYSTEM_PROMPT,
    RERANKER_THRESHOLD,
    RERANKER_CACHE_DIR,
    RERANKER_LLM_MODEL
)
from .ioutils import IOUtils
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
        self.ranker = Ranker(model_name=RERANKER_LLM_MODEL,
                             cache_dir=RERANKER_CACHE_DIR,
                             log_level="ERROR")

        if llm not in [m.model for m in ollama.list().models]:
            print(f"Installing model '{llm}'...")
            ollama.pull(llm)

    def filter_sources(self, query: str, sources: List[str]) -> List[str]:
        passages = [
            {"id": idx, "text": text}
            for idx, text in enumerate(sources)
        ]

        rank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rank_request)

        return [doc["text"] for doc in reversed(reranked_results)
                    if doc["score"] >= RERANKER_THRESHOLD]

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

            context = self.filter_sources(entry.question, sources)
            if not context:
                return MinimalAnswer(**entry.model_dump(),
                                     answer=LLM_FAILURE_ANSWER)

            user_content = (
                f"# Instructions: {LLM_SYSTEM_PROMPT}\n\n"
                f"# Context:\n{"\n\n\n".join(context)}\n\n"
                f"# Question: {entry.question}\n\n"
                f"# Remember: {LLM_SYSTEM_PROMPT}\n\n"
                "# Answer the question now:\n"
            )
            response = await self.client.chat(
                model=self.llm,
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
