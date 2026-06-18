from typing import List

# Assuming your base class is saved in src/LLMInterface.py
from src.LLMInterface import LLMInterface


class QueryRewriter(LLMInterface):
    """Local, high-speed query expansion pipeline optimized for source code and documentation."""

    def __init__(self, model: str = "gemma2:2b"):
        # Fixed the super() call syntax from super().init to super().__init__
        super().__init__(model)

    async def rewrite_query(self, query: str) -> List[str]:
        """
        Expands a user query into multiple variations using the local LLM.

        Args:
            query (str): The original user search query.

        Returns:
            List[str]: A list of unique expanded queries, including the original one.
        """
        # 1. Ensure the model is downloaded and ready
        await self._check_ready()

        # 2. Setup a strict prompt to ensure raw, fast output without conversational filler
        system_prompt = (
            "You are a precise search assistant. Break down the user's query into "
            "3 distinct search variations (using alternative technical terms or synonyms). "
            "Output ONLY the new queries, one per line. Do not include numbers, bullet points, "
            "introductory text, or explanations."
        )

        try:
            # 3. Call the async client using the inherited self.client
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                options={
                    "temperature": 0.2,   # Low temperature for consistent, focused variations
                    "num_predict": 60     # Hard limit on output tokens to maximize raw speed
                }
            )

            # 4. Extract the text response
            raw_output = response['message']['content']

            # 5. Clean, split by line, and filter out any stray empty lines
            expanded_queries = [
                line.strip()
                for line in raw_output.strip().split("\n")
                if line.strip()
            ]

            # 6. Always keep the original query to anchor your RAG vector search
            if query not in expanded_queries:
                expanded_queries.insert(0, query)

            return expanded_queries

        except Exception as e:
            print(f"Query expansion failed: {e}. Falling back to original query.")
            return [query]