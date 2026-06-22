import bm25s
import re

from pathlib import Path
from typing import List, Dict, Any

from .models import CodeChunk


class BM25Pipeline:
    def __init__(self, index_dir: str = ".") -> None:
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
