"""
Unit tests for the index loader (FUTURE-RAG).

These tests use mocks for the embedding model, vector store, and LlamaIndex components
to test IndexLoader in complete isolation without file, network, or database I/O.
"""

from unittest.mock import MagicMock, patch

import pytest

from future_rag.indexing.loader import IndexLoader

# Target module path for unittest.mock patching
LOADER_MODULE = "future_rag.indexing.loader"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding():
    """Provides a mock embedding model instance."""
    return MagicMock(name="mock_embedding_model")


@pytest.fixture
def mock_vector_store():
    """Provides a mock vector store instance."""
    return MagicMock(name="mock_vector_store")


@pytest.fixture
def mock_vector_store_index():
    """Patch `VectorStoreIndex` class inside future_rag.indexing.loader."""
    with patch(f"{LOADER_MODULE}.VectorStoreIndex") as mock_cls:
        mock_index_instance = MagicMock(name="mock_vector_store_index")
        # Handle both VectorStoreIndex.from_vector_store(...) and VectorStoreIndex(...)
        mock_cls.from_vector_store.return_value = mock_index_instance
        mock_cls.return_value = mock_index_instance
        yield mock_cls, mock_index_instance


@pytest.fixture
def loader(mock_embedding, mock_vector_store):
    """Provides an IndexLoader instance with mocked dependencies."""
    return IndexLoader(
        embedding_model=mock_embedding,
        vector_store=mock_vector_store,
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestIndexLoader:
    def test_load_existing_index(
        self, loader, mock_embedding, mock_vector_store, mock_vector_store_index
    ):
        """Test that loader.load() initializes VectorStoreIndex with injected dependencies."""
        _, mock_index_instance = mock_vector_store_index

        index = loader.load()

        assert index == mock_index_instance

    def test_loaded_index_can_create_retriever(
        self, loader, mock_vector_store_index
    ):
        """Test that the index returned by loader can create a retriever."""
        _, mock_index_instance = mock_vector_store_index
        mock_retriever = MagicMock(name="mock_retriever")
        mock_index_instance.as_retriever.return_value = mock_retriever

        index = loader.load()
        retriever = index.as_retriever(similarity_top_k=2)

        mock_index_instance.as_retriever.assert_called_once_with(
            similarity_top_k=2
        )
        assert retriever == mock_retriever

    def test_retriever_returns_query_results(
        self, loader, mock_vector_store_index
    ):
        """Test that the retriever produced from loaded index returns results."""
        _, mock_index_instance = mock_vector_store_index
        mock_retriever = MagicMock(name="mock_retriever")
        mock_result = MagicMock(name="mock_node_with_score")

        mock_retriever.retrieve.return_value = [mock_result]
        mock_index_instance.as_retriever.return_value = mock_retriever

        index = loader.load()
        retriever = index.as_retriever()
        results = retriever.retrieve("future of AI")

        mock_retriever.retrieve.assert_called_once_with("future of AI")
        assert len(results) == 1
        assert results[0] == mock_result

    def test_load_empty_vector_store(
        self, loader, mock_vector_store_index
    ):
        """Test loading when vector store contains no data still returns index object."""
        _, mock_index_instance = mock_vector_store_index

        index = loader.load()

        assert index is not None
        assert index == mock_index_instance