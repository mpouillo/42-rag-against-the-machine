"""Module handling the base connection to the local LLM via Ollama."""

import asyncio
import ollama


class LLMInterface:
    """Base class for managing the state and interactions with the local LLM.

    Attributes:
        model (str): Name of the LLM model to be used.
        client (ollama.AsyncClient): The asynchronous Ollama client instance.
    """

    def __init__(
        self,
        model: str = "qwen3:0.6b"
    ) -> None:
        """
        Initialize the LLM client and setup threading locks for model pulling.

        Args:
            model (str): Name of the model to use. Defaults to "qwen3:0.6b".
        """
        self.model = model
        self.client = ollama.AsyncClient()
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _check_ready(
        self
    ) -> None:
        """
        Check if the designated model is available locally,
        pulling it if necessary.

        This method acts as a safeguard before executing generation tasks,
        ensuring the requested model is downloaded and ready for inference.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            local_models = await self.client.list()
            model_names = [m.model for m in local_models.models]

            if self.model not in model_names:
                print(f"Pulling model {self.model}...")
                await self.client.pull(self.model)

            self._initialized = True
