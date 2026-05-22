import json
import numpy

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Any, Dict, List

from .constants import (
    VECTOR_EMBEDDINGS,
    VECTOR_CORPUS
)
from .models import MinimalSource
from .IOUtils import IOUtils


class VectorIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.corpus = []
        self._cache = {}

    def index(self, corpus: List[MinimalSource]) -> None:
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

            if not path in self._cache:
                self._cache[path] = Path(path).read_text()
            text_corpus.append(self._cache[path][start:end])

        try:
            pool = self.model.start_multi_process_pool()
            for i in tqdm(range(0, len(text_corpus), chunk_size),
                          desc="Encoding Vectors"):
                batch_text = text_corpus[i:i + chunk_size]

                batch_embeds = self.model.encode(
                    batch_text,
                    pool=pool,
                    batch_size=64
                )
                all_embeddings.append(batch_embeds)

            self.embeddings = numpy.vstack(all_embeddings).astype('float32')

        finally:
            self.model.stop_multi_process_pool(pool)

        norms = numpy.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / numpy.where(norms == 0, 1, norms)

    def search(self, query: str, k: int) -> List[MinimalSource]:
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

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        json_corpus = [c.model_dump() for c in self.corpus]
        numpy.save(str(path / VECTOR_EMBEDDINGS), self.embeddings)
        (path / VECTOR_CORPUS).write_text(json.dumps(json_corpus))

    def load(self, input_dir: str) -> None:
        path = Path(input_dir)
        self.embeddings = numpy.load(str(path / VECTOR_EMBEDDINGS))
        raw_json = json.loads((path / VECTOR_CORPUS).read_text())
        self.corpus = [MinimalSource(**src) for src in raw_json]
