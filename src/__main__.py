#!/usr/bin/env python3

import fire
import json
import bm25s
import sys
import Stemmer
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults
from typing import List, Dict, Any
from .constants import *


class Indexer:
    def __init__(self, path_to_process: str) -> None:
        self.path_to_process = path_to_process
        self.chunked_sources = []

    def chunkify(self, language: str,
                 max_chunk_size: int = 2000) -> List[Document]:
        input_path = Path(self.path_to_process)
        filenames = []
        docs = []

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

        return text_splitter.split_documents(docs)

    def save_chunks(self, chunks: list[Document], path_to_save: str) -> None:
        path = Path(path_to_save)
        path.mkdir(parents=True, exist_ok=True)
        file = path / "chunks.json"

        chunk_list = []

        for chunk in chunks:
            file_path = chunk.metadata.get("path", "")
            start_index = chunk.metadata.get("start_index", 0)
            end_index = start_index + len(chunk.page_content)

            chunk_data = {
                "file_path": file_path,
                "first_character_index": start_index,
                "last_character_index": end_index,
                "content": chunk.page_content
            }

            chunk_list.append(chunk_data)

        file.write_text(json.dumps(chunk_list))

    def index_to_file(self, chunks: List[Document],
                      save_path: str = "data/processed") -> None:
        stemmer = Stemmer.Stemmer("english")
        sources = [c.page_content for c in chunks]
        corpus_tokens = bm25s.tokenize(
            sources, stopwords="en", stemmer=stemmer
        )
        retriever = bm25s.BM25(corpus=chunks)
        retriever.index(corpus_tokens)

        json_sources = [doc.model_dump() for doc in chunks]
        retriever.save(INDEX_PATH, corpus=json_sources)


class RagInterface:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(PATH_TO_PROCESS)

        chunks = indexer.chunkify(["python", "markdown"], max_chunk_size)
        indexer.save_chunks(chunks, CHUNK_PATH)

        indexer.index_to_file(chunks, INDEX_PATH)
        print(f"Ingestion complete! Indices saved under {INDEX_PATH}")

    def search(self, query: str, k: int = 10) -> None:
        try:
            retriever = bm25s.BM25.load(INDEX_PATH, load_corpus=True)
        except Exception:
            sys.exit("Error: no index found")

        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=k)

        retrieved_sources = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i] # noqa
            retrieved_sources.append(doc.get("source"))

        result = MinimalSearchResults(
            question_id="q1",
            question=query,
            retrieved_sources=[
                MinimalSource(**json.loads(r)) for r in retrieved_sources
            ]
        )

        search_results = StudentSearchResults(search_results=[result], k=k)

        output = Path("data/output.json")
        output.write_text(json.dumps(search_results.model_dump(), indent=4))

    def search_dataset(self) -> None:
        pass

    def answer(self, query: str, k: int = 10) -> None:
        pass

    def answer_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass


def main() -> None:
    fire.Fire(RagInterface)


if __name__ == "__main__":
    main()
