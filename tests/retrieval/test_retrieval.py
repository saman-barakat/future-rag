"""
Unit tests for the Retriever wrapper class (FUTURE-RAG).

These tests isolate the Retriever class using MagicMock to verify initialization,
parameter propagation, and delegation without needing access to actual indexes,
embeddings, or database engines.
"""

from unittest.mock import MagicMock

import pytest

# Adjust import path according to your project structure
from future_rag.retrieval.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llama_retriever():
    """Provides a mock LlamaIndex retriever instance."""
    return MagicMock(name="mock_llama_retriever")


@pytest.fixture
def mock_index(mock_llama_retriever):
    """Provides a mock VectorStoreIndex that returns the mock retriever."""
    mock_idx = MagicMock(name="mock_vector_store_index")
    mock_idx.as_retriever.return_value = mock_llama_retriever
    return mock_idx


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestRetrieverInitialization:
    def test_init_default_similarity_top_k(self, mock_index, mock_llama_retriever):
        """Verify that __init__ passes the default similarity_top_k=5 to as_retriever."""
        retriever_wrapper = Retriever(index=mock_index)

        mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
        assert retriever_wrapper.retriever == mock_llama_retriever

    def test_init_custom_similarity_top_k(self, mock_index, mock_llama_retriever):
        """Verify that __init__ respects a custom similarity_top_k parameter."""
        retriever_wrapper = Retriever(index=mock_index, similarity_top_k=10)

        mock_index.as_retriever.assert_called_once_with(similarity_top_k=10)
        assert retriever_wrapper.retriever == mock_llama_retriever


class TestRetrieverRetrieve:
    def test_retrieve_delegates_query_and_returns_nodes(
        self, mock_index, mock_llama_retriever
    ):
        """Verify that retrieve() passes the query string and returns retrieved nodes."""
        # Arrange
        mock_node_1 = MagicMock(name="mock_node_1")
        mock_node_2 = MagicMock(name="mock_node_2")
        expected_nodes = [mock_node_1, mock_node_2]
        mock_llama_retriever.retrieve.return_value = expected_nodes

        retriever_wrapper = Retriever(index=mock_index, similarity_top_k=3)
        query_str = "What is the future of AI?"

        # Act
        results = retriever_wrapper.retrieve(query_str)

        # Assert
        mock_llama_retriever.retrieve.assert_called_once_with(query_str)
        assert results == expected_nodes
        assert len(results) == 2

    def test_retrieve_handles_empty_results(
        self, mock_index, mock_llama_retriever
    ):
        """Verify that retrieve() handles empty list responses gracefully."""
        # Arrange
        mock_llama_retriever.retrieve.return_value = []
        retriever_wrapper = Retriever(index=mock_index)

        # Act
        results = retriever_wrapper.retrieve("non-matching query")

        # Assert
        mock_llama_retriever.retrieve.assert_called_once_with("non-matching query")
        assert results == []