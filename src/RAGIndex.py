from .BM25Index import BM25Index
from .Chunker import Chunker
from .VectorIndex import VectorIndex


class RAGIndex:
    def __init__(
        self,
        index_directory: str
    ):
        self.index_dir = index_directory
        self.bm25 = BM25Index()
        self.vector = VectorIndex()

    def index_and_save(
        self,
        max_chunk_size: int,
        input_dir: str,
    ):
        chunker = Chunker()
        py_docs = chunker.parse_dir(input_dir, "*.py")
        md_docs = chunker.parse_dir(input_dir, "*.md")

        docs = (chunker.chunkify(md_docs, "markdown", max_chunk_size)
                + chunker.chunkify(py_docs, "python", max_chunk_size))

        sources = chunker.convert_docs_to_sources(docs)

        self.bm25.index(sources)
        self.bm25.save(self.index_dir)
        self.vector.index(sources)
        self.vector.save(self.index_dir)
