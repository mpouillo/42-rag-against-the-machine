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

    def process_chunks(self, chunks: list[Document]) -> List[Dict[str, Any]]:
        processed_sources = []

        for chunk in chunks:
            file_path = chunk.metadata.get("path", "")
            start_index = chunk.metadata.get("start_index", 0)
            end_index = start_index + len(chunk.page_content)

            source_data = {
                    "file_path": file_path,
                    "first_character_index": start_index,
                    "last_character_index": end_index
                }

            source = {
                "source": MinimalSource(**source_data).model_dump_json(),
                "content": chunk.page_content
            }

            processed_sources.append(source)

        return processed_sources

    def index_and_save(self, sources: List[Dict[str, Any]],
                       save_path: str = "data/processed") -> None:
        stemmer = Stemmer.Stemmer("english")
        corpus = [src["content"] for src in sources]
        corpus_tokens = bm25s.tokenize(
            corpus, stopwords="en", stemmer=stemmer
        )
        retriever = bm25s.BM25(corpus=sources)
        retriever.index(corpus_tokens)
        retriever.save(INDEX_PATH, corpus=sources)


INDEX_PATH = "data/processed/bm25_index"
PATH_TO_PROCESS = "data/raw"


class RagInterface:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(PATH_TO_PROCESS)
        chunks = indexer.chunkify(["python", "markdown"], max_chunk_size)
        sources = indexer.process_chunks(chunks)
        indexer.index_and_save(sources, INDEX_PATH)

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
