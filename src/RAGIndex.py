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

        Args:
            index_directory (str): Path of output directory (index data)

        Returns:
            None: None
        """
        self.index_dir = index_directory
        self.bm25 = BM25Index()
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
        docs = chunker.parse_dir(input_dir, "*.py")
        docs += chunker.parse_dir(input_dir, "*.md")
        docs += chunker.parse_dir(input_dir, "*.txt")

        docs = chunker.chunkify(docs, max_chunk_size)

        sources = chunker.convert_docs_to_sources(docs)
        unique_srcs = list(set(sources))

        if len(unique_srcs) < 1:
            raise ValueError(
                "No chunks computed, try increasing 'max_chunk_size'"
            )

        self.bm25.index(unique_srcs)
        self.bm25.save(self.index_dir)

        self.vector.index(unique_srcs)
        self.vector.save(self.index_dir)
