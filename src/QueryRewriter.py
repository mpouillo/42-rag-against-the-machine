from typing import List

from .LLMInterface import LLMInterface


class QueryRewriter(LLMInterface):
    """Local query expansion pipeline."""

    def __init__(self, model: str = "gemma2:2b"):
        super().__init__(model)

    async def rewrite_query(self, query: str) -> List[str]:
        """
        Expands a user query into multiple variations using the local LLM.

        Args:
            query (str): The original user search query.
        Returns:
            List[str]: A list of unique expanded queries.
        """
        await self._check_ready()

        system_prompt = (
            "You are a precise search assistant. Break down the user's query "
            "into 3 distinct search variations (using alternative technical "
            "terms or synonyms). Output ONLY the new queries, one per line. "
            "Do not include numbers, bullet points, introductory text, "
            "or explanations."
        )

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 60
                }
            )

            raw_output = response['message']['content']
            expanded_queries = [
                line.strip()
                for line in raw_output.strip().split("\n")
                if line.strip()
            ]

            if query not in expanded_queries:
                expanded_queries.insert(0, query)

            return expanded_queries

        except Exception:
            return [query]  # Fallback
