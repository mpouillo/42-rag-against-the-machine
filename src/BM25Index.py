import bm25s
import json
import re
import Stemmer

from pathlib import Path
from typing import Dict, List, Optional

from .constants import BM25_CORPUS, BM25_DIRECTORY
from .models import MinimalSource


class BM25Index:
    """Indexing and search pipeline using BM25."""
    def __init__(
        self
    ) -> None:
        """
        Initialize Stemmer.

        Args:
            None: None

        Returns:
            None: None
        """
        self.stemmer: Stemmer.Stemmer = Stemmer.Stemmer("english")
        self.retriever: Optional[bm25s.BM25] = None
        self._cache: Dict[str, str] = {}

    def index(
        self,
        corpus: List[MinimalSource],
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

        paths = {chunk.file_path for chunk in corpus}
        for path in paths:
            if path not in self._cache:
                self._cache[path] = Path(path).read_text()

        text_corpus = [
            self._breakup_words(
                self._cache[c.file_path]
                [c.first_character_index:c.last_character_index]
            )
            + f"\n{c.file_path}"
            for c in corpus
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
        """
        Searches local index for data matching passed query.

        Args:
            query (str): The text to match against local index
            k (int): Size of results to return

        Returns:
            List[MinimalSource]: Top k matching results
        """
        if not self.retriever:
            raise ValueError("Index is empty. Run .index() or .load() first.")

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, _ = self.retriever.retrieve(query_tokens, k=k)

        return list(results[0].tolist())

    def save(
        self,
        save_dir: str = "data/processed"
    ) -> None:
        """
        Save local index to file.

        Args:
            save_dir (str): Path to save data to

        Returns:
            None: None
        """
        if not self.retriever or not self.retriever.corpus:
            raise ValueError("Index is empty. Run .index() or .load() first.")

        json_corpus = [c.model_dump() for c in self.retriever.corpus]
        path = Path(save_dir) / BM25_DIRECTORY

        try:
            self.retriever.corpus = None
            self.retriever.save(path)
            (path / BM25_CORPUS).write_text(json.dumps(json_corpus, indent=4))
        except Exception as e:
            raise ValueError(
                f"Error saving index to {save_dir}: "
                "Path cannot be accessed, or an error "
                "occurred while writing data to file. "
                f"Details: {e}"
            )

    def load(
        self,
        load_dir: str = "data/processed",
    ) -> None:
        """
        Load index from file.

        Args:
            load_dir (str): Path to load data from

        Returns:
            None: None
        """
        bm25_path = Path(load_dir) / BM25_DIRECTORY
        corpus_path = bm25_path / BM25_CORPUS

        try:
            self.retriever = bm25s.BM25.load(bm25_path, load_corpus=False)
            raw_json = json.loads(corpus_path.read_text())
            self.retriever.corpus = [MinimalSource(**src) for src in raw_json]
        except Exception as e:
            raise ValueError(
                f"Error loading index from {load_dir}: "
                "File does not exist, or data is not valid json, "
                "or data is not a list of MinimalSources. "
                f"Details: {e}"
            )

    def _breakup_words(self, text: str) -> str:
        broken_words: List[str] = []
        for word in text.split():
            broken_words.append(word)
            if '_' in word:
                broken_words.extend([p for p in word.split('_') if p])
            camel_parts = re.findall(r'[a-zA-Z][a-z]*', word)
            if len(camel_parts) > 1:
                broken_words.extend([p.lower() for p in camel_parts])

        return " ".join(broken_words)
