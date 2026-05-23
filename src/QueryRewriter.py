import re
import spacy

from spacy.language import Language
from spacy.tokens import Doc
from typing import List


class QueryRewriter:
    """Query rewriting pipeline to improve search engine retrieval."""
    def __init__(
        self,
        model: str = "en_core_web_sm"
    ) -> None:
        """Download and load a spaCy model."""
        try:
            self.nlp = spacy.load(model, disable=["ner", "parser"])
        except Exception:
            print(f"Downloading model '{model}'...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model, disable=["ner", "parser"])

        self.allowed_pos = {"NOUN", "PROPN", "VERB", "ADJ", "X"}

    def rewrite_query(
        self,
        query: str
    ) -> str:
        """
        Rewrite a query using spaCy.

        Returns the orginal query with keywords appended at the end.

        Keyword arguments:
        - query (str): the string to rewrite
        """
        doc: Doc = self.nlp(query)

        extracted_tokens: List[str] = []
        for token in doc:
            if token.pos_ in self.allowed_pos and not token.is_stop:
                extracted_tokens.append(token.lemma_.lower())

        expanded_tokens: List[str] = []
        for tok_str in extracted_tokens:
            expanded_tokens.append(tok_str)
            if '_' in tok_str:
                expanded_tokens.extend([p for p in tok_str.split('_') if p])

            camel_parts = re.findall(r'[a-zA-Z][a-z]*', tok_str)
            if len(camel_parts) > 1:
                expanded_tokens.extend([p.lower() for p in camel_parts])

        unique_tokens = list(dict.fromkeys(expanded_tokens))
        dense_keywords = " ".join(unique_tokens)

        return f"{query} {dense_keywords}"
