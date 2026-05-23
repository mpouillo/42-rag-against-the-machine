import json
import numpy

from model2vec import StaticModel
from pathlib import Path
from typing import Any, Dict, List

from .constants import (
    VECTOR_CORPUS,
    VECTOR_EMBEDDINGS
)
from .models import MinimalSource


class VectorIndex:
    """Indexing and search pipeline using vector embeddings."""
    def __init__(
        self,
        model: str = "minishlab/potion-retrieval-32M"
    ) -> None:
        """
        Initialize SentenceTransformer model and class variables

        Args:
            model (str): Name of the LLM model to load

        Returns:
            None: None
        """
        self.model = StaticModel.from_pretrained(model)
        self.embeddings: Any = None
        self.corpus: List[MinimalSource] = []
        self._cache: Dict[str, str] = {}

    def index(
        self,
        corpus: List[MinimalSource]
    ) -> None:
        """
        Index passed corpus into embeddings in local memory.

        Args:
            corpus (List[MinimalSource]): List of sources to index

        Returns:
            None: None
        """
        if not all(isinstance(c, MinimalSource) for c in corpus):
            raise ValueError(
                "'corpus' parameter must be a list of MinimalSource objects."
            )
        self.corpus = corpus

        paths = {chunk.file_path for chunk in corpus}
        for path in paths:
            if path not in self._cache:
                self._cache[path] = Path(path).read_text()

        text_corpus = [
            self._cache[path]
            [c.first_character_index:c.last_character_index]
            for c in corpus
        ]

        self.embeddings = self.model.encode(
            text_corpus,
            show_progress_bar=True
        ).astype('float32')

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """
        Searches local index for data matching passed query.

        Args:
            query (str): The text to match against local index
            k (int): Size of results to return

        Returns:
            List[MinimalSource]: Top k matching results
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("Index is empty. Run .index() or .load() first.")

        query_embedding = self.model.encode(query).astype('float32')
        similarities = numpy.dot(self.embeddings, query_embedding)

        if len(similarities) <= k:
            top_indices = numpy.arange(len(similarities))
        else:
            partitioned_indices = numpy.argpartition(similarities, -k)
            top_indices = partitioned_indices[-k:]

        top_indices = top_indices[numpy.argsort(-similarities[top_indices])]

        return [self.corpus[idx] for idx in top_indices]

    def save(self, save_dir: str) -> None:
        """
        Save local index to file.

        Args:
            save_dir (str): Path to save data to

        Returns:
            None: None
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        try:
            json_corpus = [c.model_dump() for c in self.corpus]
            numpy.save(str(path / VECTOR_EMBEDDINGS), self.embeddings)
            (path / VECTOR_CORPUS).write_text(json.dumps(json_corpus))
        except Exception as e:
            raise ValueError(
                f"Error saving index to {save_dir}: "
                "Path cannot be accessed, or an error "
                "occurred while writing data to file. "
                f"Details: {e}"
            )

    def load(self, load_dir: str) -> None:
        """
        Load index from file.

        Args:
            load_dir (str): Path to load data from

        Returns:
            None: None
        """
        path = Path(load_dir)
        try:
            self.embeddings = numpy.load(str(path / VECTOR_EMBEDDINGS))
            raw_json = json.loads((path / VECTOR_CORPUS).read_text())
            self.corpus = [MinimalSource(**src) for src in raw_json]
        except Exception as e:
            raise ValueError(
                f"Error loading index from {load_dir}: "
                "File does not exist, or data is not valid json, "
                "or data is not a list of MinimalSources. "
                f"Details: {e}"
            )
