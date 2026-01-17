import os

# folder for embeddings db
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# Small, fast, and understands Arabic and english
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# LLM model for generating answers
# meta-llama/Llama-3.2-1B-Instruct (smaller)
# Qwen/Qwen2.5-7B-Instruct (alternative)
# AllAI/mistral-7B-instruct-v0.1 (heavy, high quality)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")


# template to force the model to use the context
QA_SYSTEM_PROMPT = os.getenv("QA_SYSTEM_PROMPT", """
You are a helpful assistant. Your task is to answer questions based strictly on the provided context. 
If the context is in Arabic, answer in Arabic. If in English, answer in English.
Do not hallucinate. If the answer is not in the context, say "I don't know".
""")