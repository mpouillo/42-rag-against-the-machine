"""Module for managing the system's indexing processes across pipelines."""

from pathlib import Path

from .BM25Pipeline import BM25Pipeline
from .ChromaPipeline import ChromaPipeline
from .Chunker import chunk_markdown_file, chunk_python_file


class RAGIndex:
    """Orchestrates indexing of documents across different search pipelines.

    Attributes:
        bm25 (BM25Pipeline): The lexical retrieval indexing handler.
        chroma (ChromaPipeline): The semantic retrieval indexing handler.
    """

    def __init__(self, index_dir: str = "data/processed") -> None:
        """Initialize both pipelines with a unified index directory.

        Args:
            index_dir (str): Directory where index data is saved.
        """
        self.bm25 = BM25Pipeline(index_dir)
        self.chroma = ChromaPipeline(index_dir)

    def ingest_and_index(
        self,
        max_chunk_size: int,
        input_dir: str = "./data/raw/",
    ) -> None:
        """Ingest, index, and save a directory's data to the system databases.

        Args:
            max_chunk_size (int): Maximum size of output chunks.
            input_dir (str): Path of the input directory to ingest.

        Raises:
            ValueError: If the input directory is not found,
                or if no valid chunks are produced.
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

        self.bm25.index_and_save(chunks)
        self.chroma.index_and_save(chunks)
