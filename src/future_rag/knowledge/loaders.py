"""
Document loaders for FUTURE-RAG.

This module loads documents from a single file or an entire directory
using LlamaIndex.
"""

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


class DocumentLoader:
    """Load documents from files or directories."""

    @staticmethod
    def load(path: str | Path) -> list[Document]:
        """
        Load one document or all supported documents in a directory.

        Parameters
        ----------
        path : str | Path
            Path to a PDF, Markdown file, or a directory.

        Returns
        -------
        list[Document]
            List of LlamaIndex Document objects.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            reader = SimpleDirectoryReader(input_files=[str(path)])
        else:
            reader = SimpleDirectoryReader(input_dir=str(path))

        return reader.load_data()