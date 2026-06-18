from pathlib import Path

from .BM25Index import bm25_index_and_save
from .Chunker import chunk_markdown_file, chunk_python_file
from .ChromaIndex import chroma_index_and_save


def ingest_and_index(
    max_chunk_size: int,
    input_dir: str = "./data/raw/",
    index_dir: str = "./data/processed/"
) -> None:
    """
    Ingest, index and save to file a directory's data.

    Args:
        max_chunk_size (int): Maximum size of output chunks
        input_dir (str): Path of input directory to ingest

    Returns:
        None: None
    """
    path = Path(input_dir)
    if not path.is_dir():
        raise ValueError("Input directory not found")

    chunks = []

    for file in path.rglob("*.py"):
        if file.is_file():
            chunks += chunk_python_file(
                str(file), file.read_text(), max_chunk_size
            )

    for file in path.rglob("*.md"):
        if file.is_file():
            chunks += chunk_markdown_file(
                str(file), file.read_text(), max_chunk_size
            )

    for file in path.rglob("*.txt"):
        if file.is_file():
            chunks += chunk_markdown_file(
                str(file), file.read_text(), max_chunk_size
            )

    if len(chunks) <= 0:
        raise ValueError(
            "No chunks computed, try increasing 'max_chunk_size'"
        )

    bm25_index_and_save(chunks, index_dir)
    chroma_index_and_save(chunks, index_dir)
