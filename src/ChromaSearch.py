# chroma_searcher.py
from pathlib import Path
from typing import List, Dict, Any, cast
import chromadb
from sentence_transformers import SentenceTransformer


class ChromaSearcher:
    def __init__(self, persist_dir: str = ".") -> None:
        path = Path(persist_dir) / "chromadb_index"
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_collection(name="chromadb_index")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            trust_remote_code=True
        )

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results: chromadb.QueryResult = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )

        formatted_results = []
        if (
            results and results["metadatas"]
            and results["documents"] and results["distances"]
        ):
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for meta, doc_text, dist in zip(metadatas, documents, distances):
                score = 1.0 - float(dist)
                formatted_results.append({
                    "file_path": meta["file_path"],
                    "first_character_index": int(cast(
                        Any, meta.get("first_character_index")
                    )),
                    "last_character_index": int(cast(
                        Any, meta.get("last_character_index")
                    )),
                    "text": doc_text,
                    "score": score
                })
        return formatted_results

    def search_batch(
        self,
        queries: List[str],
        k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        query_vectors = self.model.encode(
            queries, batch_size=64, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_vectors,
            n_results=k
        )

        batch_output = []
        for i in range(len(queries)):
            query_results = []
            if (
                results and results["metadatas"]
                and results["documents"] and results["distances"]
            ):
                metadatas = results["metadatas"][i]
                documents = results["documents"][i]
                distances = results["distances"][i]

                for meta, doc_text, dist in zip(
                    metadatas, documents, distances
                ):
                    score = 1.0 - float(dist)
                    query_results.append({
                        "file_path": meta["file_path"],
                        "first_character_index": int(cast(
                            Any, meta.get("first_character_index")
                        )),
                        "last_character_index": int(cast(
                            Any, meta.get("last_character_index")
                        )),
                        "text": doc_text,
                        "score": score
                    })
            batch_output.append(query_results)
        return batch_output
