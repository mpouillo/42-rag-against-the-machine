import bm25s
import json
import Stemmer

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from typing import Any, Dict, List

from .models import MinimalSource


class Indexer:
    def __init__(self, path_to_process: str) -> None:
        self.path_to_process = path_to_process
        self.chunks_filename = "chunks.json"

    def chunkify(self, language: str,
                 max_chunk_size: int = 2000) -> List[Document]:
        input_path = Path(self.path_to_process)
        filenames = []
        docs = []
        chunks = []

        match language:
            case "python":
                lang = Language.PYTHON
            case _:
                lang = Language.MARKDOWN

        for filename in input_path.rglob('*'):
            if filename.is_file() and filename.suffix == '.py':
                filenames.append(filename)

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=max_chunk_size,
            chunk_overlap=0,    # maybe more?
            add_start_index=True,
            keep_separator=True
        )

        for file in filenames:
            docs.append(
                Document(page_content=file.read_text(),
                         metadata={"path": str(file)})
            )

        split_docs = text_splitter.split_documents(docs)

        for doc in split_docs:
            file_path = doc.metadata.get("path", "")
            start_index = doc.metadata.get("start_index", 0)
            end_index = start_index + len(doc.page_content)

            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start_index,
                    last_character_index=end_index
                )
            )

        return chunks

    @staticmethod
    def save_chunks(chunks: list[Document], path_to_save: str) -> None:
        path = Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)

        chunk_list = [chunk.model_dump() for chunk in chunks]
        path.write_text(json.dumps(chunk_list, indent=4))

    @staticmethod
    def load_chunks(path_to_load: str) -> Dict[str, Any]:
        path = Path(path_to_load)

        return [MinimalSource(**obj) for obj in json.loads(path.read_text())]

    def create_index(self, chunks: List[Document],
                     save_path: str = "data/processed") -> None:
        stemmer = Stemmer.Stemmer("english")
        text_srcs = []

        for chunk in chunks:
            file_path = chunk.file_path
            start_index = chunk.first_character_index
            end_index = chunk.last_character_index
            file = Path(file_path)

            text_srcs.append(file.read_text()[start_index:end_index])

        corpus_tokens = bm25s.tokenize(
            text_srcs, stopwords="en", stemmer=stemmer
        )
        retriever = bm25s.BM25(corpus=chunks)
        retriever.index(corpus_tokens)

        json_chunks = [c.model_dump() for c in chunks]

        retriever.save(save_path, corpus=json_chunks)
