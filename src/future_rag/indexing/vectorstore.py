"""
Vector store for FUTURE-RAG.

This module creates or connects to a persistent ChromaDB vector store.
"""

from pathlib import Path

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


class VectorStore:
    """Manage the ChromaDB vector store."""

    def __init__(
        self,
        db_path: str | Path = "./data/vectorstores/chroma",
        collection_name: str = "future_rag",
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name

    def connect(self) -> ChromaVectorStore:
        """
        Create or connect to the ChromaDB collection.

        Returns
        -------
        ChromaVectorStore
            LlamaIndex Chroma vector store.
        """

        self.db_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        collection = client.get_or_create_collection(
            name=self.collection_name
        )

        return ChromaVectorStore(
            chroma_collection=collection
        )