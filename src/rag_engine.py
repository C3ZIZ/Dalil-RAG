"""
Docstring for src.rag_engine

1- vector store with chroma
2- embedding
3- Retrieve the relevant text chunks and sends them to the LLM.
"""

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from src.config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    QA_SYSTEM_PROMPT,
)


class RAGEngine:
    def __init__(self, hf_token):
        self.hf_token = hf_token
        self._initialize_settings()
        self.index = None

    def _initialize_settings(self):
        # embedding model local
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

        # llm model using HuggingFace Inference API
        Settings.llm = OpenAILike(
            model=LLM_MODEL_NAME,
            api_base="https://router.huggingface.co/v1/",
            api_key=self.hf_token,
            is_chat_model=True,
            context_window=4096,
            max_tokens=512,
            temperature=0.2,
        )

    def build_index(self, documents):
        db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        chroma_collection = db.get_or_create_collection("quick_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # chunk and create embedding
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        return self.index

    def get_query_engine(self):
        """Returns the engine that can answer questions"""
        if not self.index:
            return None

        # Create a query engine with the custom system prompt
        return self.index.as_query_engine(
            streaming=True,
            similarity_top_k=3,# top 3 relevant chunks
        )
