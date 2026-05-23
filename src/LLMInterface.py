import ollama


class LLMInterface:
    def __init__(
        self,
        model: str = "qwen3:0.6b"
    ) -> None:
        """
        Initialize LLM client.

        Args:
            model (str): Name of the model to use

        Returns:
            None: None
        """
        self.model = model
        self.client = ollama.AsyncClient()
        self._initialized = False

    async def _check_ready(
        self
    ) -> None:
        """Helper function to pull requested model if needed."""

        if self._initialized:
            return

        local_models = await self.client.list()
        model_names = [m.model for m in local_models.models]

        if self.model not in model_names:
            print(f"Pulling model {self.model}...")
            await self.client.pull(self.model)

        self._initialized = True
