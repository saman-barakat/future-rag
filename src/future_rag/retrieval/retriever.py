"""
Retriever for FUTURE-RAG.
"""

from llama_index.core.indices.vector_store import VectorStoreIndex


class Retriever:
    """
    Wrapper around the LlamaIndex retriever.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 5,
    ):
        """
        Parameters
        ----------
        index : VectorStoreIndex
            Built vector index.

        similarity_top_k : int
            Number of retrieved chunks.
        """

        self.retriever = index.as_retriever(
            similarity_top_k=similarity_top_k
        )

    def retrieve(self, query: str):
        """
        Retrieve the most relevant nodes.

        Parameters
        ----------
        query : str

        Returns
        -------
        list
            Retrieved nodes.
        """

        return self.retriever.retrieve(query)