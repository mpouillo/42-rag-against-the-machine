import os

from .BM25Index import BM25Index
from .Chunker import Chunker
from .constants import MODEL_VECTOR
from .VectorIndex import VectorIndex


class RAGIndex:
    """Core RAG class used to ingest and index a directory of files."""
    def __init__(
        self,
        index_directory: str
    ) -> None:
        """
        Initialize BM25 and Vector databases.
        If environment variable "HYBRID_RETRIEVAL" is set to True,
        uses hybrid retrieval for search computation.

        Args:
            index_directory (str): Path of output directory (index data)

        Returns:
            None: None
        """
        self.index_dir = index_directory
        self.hybrid_retrieval = (
            True if os.environ.get("HYBRID_RETRIEVAL", False) in ["True", True]
            else False
        )
        self.bm25 = BM25Index()

        if self.hybrid_retrieval:
            self.vector = VectorIndex(MODEL_VECTOR)

    def index_and_save(
        self,
        max_chunk_size: int,
        input_dir: str,
    ) -> None:
        """
        Ingest, index and save to file a directory's data.

        Args:
            max_chunk_size (int): Maximum size of output chunks
            input_dir (str): Path of input directory to ingest

        Returns:
            None: None
        """
        chunker = Chunker()
        py_docs = chunker.parse_dir(input_dir, "*.py")
        md_docs = chunker.parse_dir(input_dir, "*.md")

        docs = (chunker.chunkify(md_docs, "markdown", max_chunk_size)
                + chunker.chunkify(py_docs, "python", max_chunk_size))

        sources = chunker.convert_docs_to_sources(docs)
        unique_srcs = list(set(sources))

        self.bm25.index(unique_srcs)
        self.bm25.save(self.index_dir)

        if self.hybrid_retrieval:
            self.vector.index(unique_srcs)
            self.vector.save(self.index_dir)
