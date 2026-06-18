from pathlib import Path

import chromadb

from tqdm import tqdm
from typing import List, Any, cast
from sentence_transformers import SentenceTransformer


def chroma_index_and_save(
    chunks: List[Any],
    persist_dir: str = "."
) -> None:
    path = Path(persist_dir) / "chromadb_index"
    path.parent.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name="chromadb_index",
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        trust_remote_code=True
    )

    batch_size = 512
    total_chunks = len(chunks)

    for i in tqdm(
        range(0, total_chunks, batch_size),
        desc="Embedding and Ingesting Batches"
    ):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]

        batch_texts = [chunk.text for chunk in batch_chunks]

        batch_metadatas = []
        batch_ids = []
        for idx, chunk in enumerate(batch_chunks):
            batch_metadatas.append(cast(Any, {
                "file_path": chunk.source.file_path,
                "first_character_index": chunk.source.first_character_index,
                "last_character_index": chunk.source.last_character_index
            }))
            batch_ids.append(f"doc_chunk_{i + idx}")

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadatas
        )
