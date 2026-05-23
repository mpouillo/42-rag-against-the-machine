import json
import numpy

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Any, Dict, List

from .constants import VECTOR_CORPUS, VECTOR_EMBEDDINGS
from .models import MinimalSource


class VectorIndex:
    """Indexing and search pipeline using vector embeddings."""
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize SentenceTransformer model and class variables

        Args:
            model (str): Name of the LLM model to load

        Returns:
            None: None
        """
        self.model: SentenceTransformer = SentenceTransformer(model)
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
                "'corpus' parameter must be a list of MinimalSource."
            )
        self.corpus = corpus
        all_embeddings = []
        chunk_size = 2000

        text_corpus = []
        for chunk in corpus:
            path = chunk.file_path
            start = chunk.first_character_index
            end = chunk.last_character_index

            if path not in self._cache:
                self._cache[path] = Path(path).read_text()
            text_corpus.append(self._cache[path][start:end])

        for i in tqdm(range(0, len(text_corpus), chunk_size), desc="Encoding vectors"):
            batch_text = text_corpus[i:i + chunk_size]

            batch_embeds = self.model.encode(
                batch_text,
                batch_size=64,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeds)

        self.embeddings = numpy.vstack(all_embeddings).astype('float32')

        norms = numpy.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / numpy.where(norms == 0, 1, norms)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """
        Searches local index for data matching passed query.

        Args:
            query (str): The text to match against local index
            k (int): Size of results to return

        Returns:
            List[MinimalSource]: Top k matching results
        """
        if self.embeddings is None:
            raise ValueError("Index is empty. Run .index() or .load() first.")

        query_vector = self.model.encode(
            query, convert_to_numpy=True
        ).astype('float32')
        query_norm = numpy.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        scores = numpy.dot(self.embeddings, query_vector)
        top_indices = numpy.argsort(scores)[::-1][:k]

        return [self.corpus[i] for i in top_indices]

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
