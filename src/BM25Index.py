import bm25s
import json
import Stemmer

from pathlib import Path
from typing import List

from .constants import BM25_DIRECTORY, BM25_CORPUS
from .models import MinimalSource
from .IOUtils import IOUtils


class BM25Index:
    def __init__(
        self
    ) -> None:
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = None

    def index(
        self,
        corpus: List[MinimalSource],
    ) -> None:
        if not all(isinstance(c, MinimalSource) for c in corpus):
            raise ValueError(
                "'corpus' parameter must be a list of MinimalSource."
            )

        text_corpus = [
            f"[Source file: {chunk.file_path}]\n" +
            IOUtils.get_text_from_file(**chunk.model_dump())
            for chunk in corpus
        ]

        corpus_tokens = bm25s.tokenize(
            text_corpus, stopwords="en", stemmer=self.stemmer, leave=True
        )

        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(corpus_tokens)

    def search(
        self,
        query: str,
        k: int
    ) -> List[MinimalSource]:
        if not self.retriever:
            raise ValueError("Index is empty. Run .index() or .load() first.")

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, _ = self.retriever.retrieve(query_tokens, k=k)

        return results[0].tolist()

    def save(
        self,
        save_dir: str = "data/processed"
    ) -> None:
        if not self.retriever:
            raise ValueError("Index is empty. Run .index() or .load() first.")

        json_corpus = [c.model_dump() for c in self.retriever.corpus]
        path = Path(save_dir) / BM25_DIRECTORY
        self.retriever.corpus = None
        self.retriever.save(path)
        (path / BM25_CORPUS).write_text(json.dumps(json_corpus, indent=4))

    def load(
        self,
        load_dir: str = "data/processed",
    ) -> None:
        bm25_path = Path(load_dir) / BM25_DIRECTORY
        corpus_path = bm25_path / BM25_CORPUS
        self.retriever = bm25s.BM25.load(bm25_path, load_corpus=False)
        raw_json = json.loads(corpus_path.read_text())
        self.retriever.corpus = [MinimalSource(**src) for src in raw_json]
