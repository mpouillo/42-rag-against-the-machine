# Files
INDEX_DIRECTORY = "data/processed/"
BM25_DIRECTORY = "bm25"
INGEST_DIRECTORY = "data/raw"
SEARCH_DIRECTORY = "data/output/search_results"
BM25_CORPUS = "metadata_corpus.json"

VECTOR_DIRECTORY = "vector"
VECTOR_CORPUS = "metadata_corpus.json"
VECTOR_EMBEDDINGS = "vectors.npy"


# LLM
LLM_TEMPERATURE = 0.2
LLM_NUM_PREDICT = 1024
LLM_FAILURE_ANSWER = "I'm sorry, but I couldn't find any relevant information \
in the database to answer your question."
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
user.
4. Omit any thinking (e.g., <think> tags) or reasoning (e.g., "Based on...").
5. Keep the response clear, precise, natural, and under 5 sentences.
"""

# Reranker
RERANKER_THRESHOLD = 0.5
RERANKER_CACHE_DIR = "./opt"
RERANKER_LLM_MODEL = "ms-marco-MiniLM-L-12-v2"
# Faster: "ms-marco-TinyBERT-L-2-v2"

# Recall@k
RECALL_THRESHOLD = 0.05

VECTOR_SEARCH_LLM = "all-MiniLM-L6-v2"
