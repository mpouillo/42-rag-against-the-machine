"""ChromaDB indexing and semantic search pipeline."""
# chroma_searcher.py
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, cast
from tqdm import tqdm


class ChromaPipeline:
    """
    Manages the creation, saving, and semantic searching of a ChromaDB index.

    Attributes:
        path (Path): The file path where the ChromaDB index is persisted.
        client (chromadb.PersistentClient): The persistent ChromaDB client.
        collection (chromadb.Collection): The collection storing embeddings.
        model (SentenceTransformer): The embedding model used.
    """

    def __init__(self, persist_dir: str = ".") -> None:
        """Initialize the ChromaDB pipeline and embedding model.

        Args:
            persist_dir (str): The root directory to persist the database.
        """
        self.path = Path(persist_dir) / "chromadb_index"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name="chromadb_index",
            metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def index_and_save(
        self,
        chunks: List[Any],
    ) -> None:
        """Embed and index a list of document chunks into ChromaDB in batches.

        Args:
            chunks (List[Any]): Chunked documents to be embedded and indexed.
        """
        batch_size = 512
        total_chunks = len(chunks)

        for i in tqdm(
            range(0, total_chunks, batch_size),
            desc="Indexing with ChromaDB..."
        ):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]

            batch_texts = [chunk.text for chunk in batch_chunks]

            batch_metadatas = []
            batch_ids = []
            for idx, chunk in enumerate(batch_chunks):
                batch_metadatas.append(cast(Any, {
                    "file_path": chunk.source.file_path,
                    "first_character_index": (
                        chunk.source.first_character_index
                    ),
                    "last_character_index": chunk.source.last_character_index
                }))
                batch_ids.append(f"doc_chunk_{i + idx}")

            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadatas
            )

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the top-k most semantically relevant chunks for a query.

        Args:
            query (str): The search query string.
            k (int): The number of top results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: Formatted search results.
        """
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
