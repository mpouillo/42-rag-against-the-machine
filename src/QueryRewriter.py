import spacy
import re

class QueryRewriter:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        # Load the small, ultra-fast English model
        # Disable Named Entity Recognition (ner) and Parser (parser) to maximize processing speed
        self.nlp = spacy.load(model_name, disable=["ner", "parser"])

        # Allowed POS tags that contain high-signal content
        self.allowed_pos = {"NOUN", "PROPN", "VERB", "ADJ", "X"}

    def rewrite_query(self, query: str) -> str:
        # 1. Process the query string through spaCy
        doc = self.nlp(query)

        extracted_tokens = []
        for token in doc:
            # 2. Filter by Part-of-Speech and ensure it's not a standard stopword
            if token.pos_ in self.allowed_pos and not token.is_stop:
                # Use the lemma_ (root form) for verbs/nouns to increase BM25 matching flexibility
                extracted_tokens.append(token.lemma_.lower())

        # 3. Code-Specific Tokenization Fallback:
        # Since spaCy is trained on normal English text, it handles variable names
        # like 'max_num_seqs' as single tokens. We manually split snake_case/camelCase terms.
        expanded_tokens = []
        for token in extracted_tokens:
            expanded_tokens.append(token)
            if '_' in token:
                expanded_tokens.extend([p for p in token.split('_') if p])

            # Match camelCase components
            camel_parts = re.findall(r'[a-zA-Z][a-z]*', token)
            if len(camel_parts) > 1:
                expanded_tokens.extend([p.lower() for p in camel_parts])

        # 4. Deduplicate tokens while maintaining their original order
        unique_tokens = list(dict.fromkeys(expanded_tokens))
        dense_keywords = " ".join(unique_tokens)

        # Combine original query with the dense semantic keywords
        return f"{query} {dense_keywords}"
