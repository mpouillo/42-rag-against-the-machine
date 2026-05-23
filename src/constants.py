"""Constant variables for easy tweaking."""

# Path
BM25_CORPUS = "metadata_corpus.json"
BM25_DIRECTORY = "bm25"
INDEX_DIRECTORY = "data/processed/"
INGEST_DIRECTORY = "data/raw/"
SEARCH_DIRECTORY = "data/output/search_results"
VECTOR_CORPUS = "metadata_corpus.json"
VECTOR_EMBEDDINGS = "vectors.npy"

# Answrer
LLM_CONTEXT_TRIM = 3
LLM_FAILURE_ANSWER = "I'm sorry, but I couldn't find any relevant information \
in the database to answer your question."
LLM_NUM_PREDICT = 1024
LLM_SYSTEM_PROMPT = """
You are a technical documentation assistant. Your job is to answer the \
user's question directly and factually using the provided context.

Follow these strict output rules:
1. State the answer immediately as a direct, standalone declarative statement.\
 Always omit intro phrases or follow-up questions.
2. The user must be able to understand what the original question was just by \
reading your answer.
3. Never reference or mention the context directly. Act as if the information \
provided in the context is your own permanent knowledge and explain it to the \
user while citing the source.
4. Omit any thinking (e.g., <think> tags) or reasoning (e.g., "Based on...").
5. Keep the response clear, precise, natural, and under 5 sentences.
"""
LLM_TEMPERATURE = 0.2

# Misc
PROMPT_CROP = 2000

# Reranker
RERANKER_CACHE_DIR = "./opt"
RERANKER_THRESHOLD = 0.5

# Recall@k
RECALL_THRESHOLD = 0.05

# Models
MODEL_ANSWER = "qwen3:0.6b"
MODEL_RERANKER = "ms-marco-MiniLM-L-12-v2"
# Alt: "ms-marco-TinyBERT-L-2-v2"
MODEL_REWRITER = "en_core_web_sm"
MODEL_VECTOR = "minishlab/potion-retrieval-32M"
# Alt: "minishlab/potion-base-8M"
