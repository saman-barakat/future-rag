"""
Document chunking for FUTURE-RAG.

This module converts LlamaIndex Documents into Nodes using the
SentenceSplitter.
"""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode


class DocumentChunker:
    """Split documents into chunks (nodes)."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
    ):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[BaseNode]:
        """
        Split documents into nodes.

        Parameters
        ----------
        documents : list[Document]
            Documents loaded by DocumentLoader.

        Returns
        -------
        list[BaseNode]
            Chunked nodes.
        """
        return self.splitter.get_nodes_from_documents(documents)