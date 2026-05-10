#!/usr/bin/env python3

import fire
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .models import MinimalSource


class Chunker:
    def process(self, max_chunk_size: int = 2000):
        path = Path("data/raw")

        filenames = []
        docs = []
        sources = []

        for filename in path.rglob('*'):
            if filename.is_file() and filename.suffix == '.py':
                filenames.append(filename)

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True,
            keep_separator=True
        )

        for file in filenames:
            docs.append(
                Document(page_content=file.read_text(),
                         metadata={"source": str(file)})
            )

        for chunk in text_splitter.split_documents(docs):
            file_path = chunk.metadata.get("source")
            start_index = chunk.metadata.get("start_index")
            end_index = start_index + len(chunk.page_content)

            chunk = {
                    "file_path": file_path,
                    "first_character_index": start_index,
                    "last_character_index": end_index
                }

            MinimalSource(**chunk)
            sources.append(chunk)

        return sources


class RagInterface:
    def index(self, max_chunk_size: int = 2000) -> None:
        sources = Chunker.process(max_chunk_size)
        file = Path("data/sources.json")
        file.write_text(json.dumps(sources, indent=4))

    def search(self) -> None:
        print("test")

    def search_dataset(self) -> None:
        print("test")

    def answer(self) -> None:
        print("test")

    def answer_dataset(self) -> None:
        print("test")

    def evaluate(self) -> None:
        print("test")


def main() -> None:
    fire.Fire(RagInterface)


if __name__ == "__main__":
    main()
