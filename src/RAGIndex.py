from pathlib import Path

from .BM25Pipeline import BM25Pipeline
from .ChromaPipeline import ChromaPipeline
from .Chunker import chunk_markdown_file, chunk_python_file


class RAGIndex:
    def __init__(self, index_dir: str = "data/processed") -> None:
        self.bm25 = BM25Pipeline(index_dir)
        self.chroma = ChromaPipeline(index_dir)

    def ingest_and_index(
        self,
        max_chunk_size: int,
        input_dir: str = "./data/raw/",
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

        self.bm25.index_and_save(chunks)
        self.chroma.index_and_save(chunks)
