"""
Index loader for FUTURE-RAG.
"""

from llama_index.core import VectorStoreIndex


class IndexLoader:
    """
    Load an existing VectorStoreIndex.
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def load(self) -> VectorStoreIndex:
        """
        Load an existing vector index.

        Returns
        -------
        VectorStoreIndex
        """

        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embedding_model,
        )