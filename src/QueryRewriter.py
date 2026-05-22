import spacy
import re

class QueryRewriter:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model_name, disable=["ner", "parser"])
        self.allowed_pos = {"NOUN", "PROPN", "VERB", "ADJ", "X"}

    def rewrite_query(self, query: str) -> str:
        doc = self.nlp(query)

        extracted_tokens = []
        for token in doc:
            if token.pos_ in self.allowed_pos and not token.is_stop:
                extracted_tokens.append(token.lemma_.lower())

        expanded_tokens = []
        for token in extracted_tokens:
            expanded_tokens.append(token)
            if '_' in token:
                expanded_tokens.extend([p for p in token.split('_') if p])

            camel_parts = re.findall(r'[a-zA-Z][a-z]*', token)
            if len(camel_parts) > 1:
                expanded_tokens.extend([p.lower() for p in camel_parts])

        unique_tokens = list(dict.fromkeys(expanded_tokens))
        dense_keywords = " ".join(unique_tokens)

        return f"{query} {dense_keywords}"
