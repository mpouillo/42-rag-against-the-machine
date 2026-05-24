from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List

from .models import MinimalSource


class Chunker:
    """Class enabling parsing of text chunks into MinimalSource objects."""
    def chunkify(
        self,
        docs: List[Document],
        max_chunk_size: int = 2000
    ) -> List[Document]:
        """
        Split a list of Document objects into chunks of maximum
        max_chunk_size characters, depending on data language.

        Args:
            docs (List[Document]): List of Document objects to be split
            language (str): Language of docs
            max_chunk_size (int): Maximum size of output chunks

        Returns:
            List[Document]: list of smaller chunks (< max_chunk_size)
        """
        if max_chunk_size <= 0:
            raise ValueError(
                "'max_chunk_size' must be a positive non-zero integer"
            )

        split_docs = []
        while max_chunk_size >= 200:
            split_docs += RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=max_chunk_size,
                    chunk_overlap=max_chunk_size // 4,
                    add_start_index=True,
                    keep_separator=True
                ).split_documents(docs)
            max_chunk_size = max_chunk_size // 2

        if not split_docs:
            split_docs += RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=max_chunk_size,
                    chunk_overlap=max_chunk_size // 4,
                    add_start_index=True,
                    keep_separator=True
                ).split_documents(docs)

        return split_docs

    @staticmethod
    def parse_dir(
        dir_path: str,
        pattern: str
    ) -> List[Document]:
        """
        Read all files in a directory matching a pattern
        and return them as a list of Document objects.

        Args:
            dir_path (str): Path of directory to ingest
            pattern (str): Pattern of files to ingest

        Returns:
            List[Document]: list of files parsed as Document objects.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError("Input directory not found")

        if not list(path.rglob(pattern)):
            raise ValueError("No files to ingest")

        return [
            Document(page_content=file.read_text(),
                     metadata={"path": str(file)})
            for file in path.rglob(pattern) if file.is_file()
        ]

    @staticmethod
    def convert_docs_to_sources(
        docs: List[Document]
    ) -> List[MinimalSource]:
        """
        Convert a list of Document objects to a list of MinimalSource objects.

        Args:
            docs (List[Document]): List of Document objects to be converted

        Return:
            List[MinimalSource]: List of converted MinimalSource objects
        """
        sources = []
        for doc in docs:
            file_path = doc.metadata.get("path", "")
            start = doc.metadata.get("start_index", 0)
            end = start + len(doc.page_content)

            sources.append(MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            ))

        return sources
