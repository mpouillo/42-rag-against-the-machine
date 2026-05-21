from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from typing import List

from .models import MinimalSource


class Chunker:
    @staticmethod
    def parse_dir(
        dir_path: str,
        pattern: str
    ) -> List[Document]:
        """Return a list of files in a directory matching a pattern"""
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError("input directory not found")
        if not path.rglob('*'):
            raise ValueError("no files to ingest")

        return [
            Document(page_content=file.read_text(),
                     metadata={"path": str(file)})
            for file in path.rglob(pattern) if file.is_file()
        ]

    @staticmethod
    def chunkify(
        docs: List[Document],
        language: str,
        max_chunk_size: int = 2000
    ) -> List[Document]:
        match language:
            case "python":
                lang = Language.PYTHON
            case _:
                lang = Language.MARKDOWN

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=max_chunk_size,
            chunk_overlap=max(150, max_chunk_size // 5),
            add_start_index=True,
            keep_separator=True
        )

        return text_splitter.split_documents(docs)

    @staticmethod
    def convert_docs_to_sources(
        docs: List[Document]
    ) -> List[MinimalSource]:
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
