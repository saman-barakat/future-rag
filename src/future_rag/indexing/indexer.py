"""
Index builder for FUTURE-RAG.
"""

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import BaseNode


class IndexBuilder:
    """
    Build a LlamaIndex VectorStoreIndex.
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def build(
        self,
        nodes: list[BaseNode],
    ) -> VectorStoreIndex:
        """
        Build a vector index from document nodes.
        """

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embedding_model,
        )