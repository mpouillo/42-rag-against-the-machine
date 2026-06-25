"""BM25 indexing and retrieval pipeline."""

import bm25s
import re

from pathlib import Path
from typing import List, Dict, Any

from .models import CodeChunk


class BM25Pipeline:
    """Manages the creation, saving, and searching of a BM25 document index.

    Attributes:
        path (Path): The file path where the BM25 index is stored.
        retriever (bm25s.BM25 | None): The active BM25 retriever instance.
    """

    def __init__(self, index_dir: str = ".") -> None:
        """Initialize the BM25 pipeline. Load an existing index if available.

        Args:
            index_dir (str): Root directory to store or load the index.
        """
        self.path = Path(index_dir) / "bm25_index"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.retriever = bm25s.BM25.load(str(self.path), load_corpus=True)
        except Exception:
            self.retriever = None

    def index_and_save(
        self,
        chunks: List[CodeChunk],
    ) -> None:
        """Process, tokenize, index and save to disk a list of code chunks.

        Args:
            chunks (List[CodeChunk]): List of chunked documents to be indexed.
        """
        corpus_texts = [chunk.text for chunk in chunks]

        metadata_corpus = []
        for chunk in chunks:
            metadata_corpus.append({
                "text": chunk.text,
                "file_path": chunk.source.file_path,
                "first_character_index": chunk.source.first_character_index,
                "last_character_index": chunk.source.last_character_index
            })

        corpus_tokens = [self.tokenize_code(text) for text in corpus_texts]

        retriever = bm25s.BM25(method="lucene")
        retriever.index(corpus_tokens)
        retriever.save(str(self.path), corpus=metadata_corpus)
        self.retriever = retriever

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a given query.

        Args:
            query (str): The search query string.
            k (int): The number of top results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: Formatted results.

        Raises:
            FileNotFoundError: If search is attempted before an index exists.
        """
        if not self.retriever:
            raise FileNotFoundError("No index found. Run '.index' first.")

        query_tokens = self.tokenize_code(query)
        results, scores = self.retriever.retrieve([query_tokens], k=k)

        formatted_results = []
        for doc, score in zip(results[0], scores[0]):
            formatted_results.append({
                "file_path": doc["file_path"],
                "first_character_index": int(doc["first_character_index"]),
                "last_character_index": int(doc["last_character_index"]),
                "text": doc["text"],
                "score": float(score)
            })
        return formatted_results

    @staticmethod
    def tokenize_code(text: str) -> List[str]:
        """Tokenize a string of code, handling snake_case and camelCase splits.

        Args:
            text (str): The raw text or code to tokenize.

        Returns:
            List[str]: A list of processed and extended tokens.
        """
        tokens = re.findall(r'\w+', text.lower())
        extended_tokens = list(tokens)

        for token in tokens:
            if '_' in token:
                extended_tokens.extend(
                    [t for t in token.split('_') if len(t) > 1]
                )
            camel_splits = re.findall(r'[a-zA-Z][^A-Z]*', token)
            if len(camel_splits) > 1:
                extended_tokens.extend([
                    t.lower() for t in camel_splits if len(t) > 1]
                )

        return extended_tokens
