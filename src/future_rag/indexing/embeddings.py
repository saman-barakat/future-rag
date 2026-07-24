"""
Embedding model factory for FUTURE-RAG.
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EmbeddingFactory:
    """Factory for creating embedding models."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    @classmethod
    def create(
        cls,
        model_name: str | None = None,
    ) -> HuggingFaceEmbedding:
        """
        Create a HuggingFace embedding model.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model name.

        Returns
        -------
        HuggingFaceEmbedding
        """

        model_name = model_name or cls.DEFAULT_MODEL

        return HuggingFaceEmbedding(
            model_name=model_name
        )