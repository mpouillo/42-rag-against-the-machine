import ollama

class LLMInterface:
    def __init__(self, llm: str = "qwen3:0.6b") -> None:
        self.llm = llm
        self.client = ollama.AsyncClient()
        self._initialized = False

    async def _check_ready(self) -> None:
        if self._initialized:
            return

        local_models = await self.client.list()
        model_names = [m.model for m in local_models.models]

        if self.llm not in model_names:
            print(f"Pulling model {self.llm}...")
            await self.client.pull(self.llm)

        self._initialized = True
