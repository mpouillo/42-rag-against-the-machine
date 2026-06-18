import bm25s
import re

from pathlib import Path
from typing import List

from .models import CodeChunk


def code_tokenizer(text: str) -> List[str]:
    tokens = re.findall(r'\w+', text.lower())
    extended_tokens = list(tokens)

    for token in tokens:
        if '_' in token:
            extended_tokens.extend(
                [t for t in token.split('_') if len(t) > 1]
            )
        camel_splits = re.findall(r'[a-zA-Z][^A-Z]*', token)
        if len(camel_splits) > 1:
            extended_tokens.extend([
                t.lower() for t in camel_splits if len(t) > 1]
            )

    return extended_tokens


def bm25_index_and_save(
    chunks: List[CodeChunk],
    save_dir: str = "./bm25_index"
) -> None:
    corpus_texts = [chunk.text for chunk in chunks]

    metadata_corpus = []
    for chunk in chunks:
        metadata_corpus.append({
            "text": chunk.text,
            "file_path": chunk.source.file_path,
            "first_character_index": chunk.source.first_character_index,
            "last_character_index": chunk.source.last_character_index
        })

    corpus_tokens = [code_tokenizer(text) for text in corpus_texts]

    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)

    path = Path(save_dir) / "bm25_index"
    path.parent.mkdir(parents=True, exist_ok=True)
    retriever.save(str(path), corpus=metadata_corpus)
