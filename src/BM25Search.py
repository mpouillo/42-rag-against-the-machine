from pathlib import Path

import bm25s

from typing import List, Dict, Any

from .BM25Index import code_tokenizer


class BM25Searcher:
    def __init__(self, index_dir: str = ".") -> None:
        path = Path(index_dir) / "bm25_index"
        self.retriever = bm25s.BM25.load(str(path), load_corpus=True)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        query_tokens = code_tokenizer(query)
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

    def search_batch(
        self,
        queries: List[str],
        k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        tokenized_queries = [code_tokenizer(q) for q in queries]
        results, scores = self.retriever.retrieve(tokenized_queries, k=k)

        batch_output = []
        for i in range(len(queries)):
            query_results = []
            for doc, score in zip(results[i], scores[i]):
                query_results.append({
                    "file_path": doc["file_path"],
                    "first_character_index": int(doc["first_character_index"]),
                    "last_character_index": int(doc["last_character_index"]),
                    "text": doc["text"],
                    "score": float(score)
                })
            batch_output.append(query_results)
        return batch_output
