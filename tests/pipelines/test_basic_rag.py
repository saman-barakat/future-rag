"""
Unit tests for the BasicRAG pipeline (FUTURE-RAG).

These tests isolate the BasicRAG class using MagicMock and patching
to verify initialization, prompt configuration, query delegation,
asynchronous query delegation, and retrieval delegation.

No real LLM calls, vector database, or retrieval operations are used.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from future_rag.pipelines.basic_rag import (
    DEFAULT_QA_TEMPLATE,
    BasicRAG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_retriever():
    """Provides a mock Retriever instance."""
    return MagicMock(name="mock_retriever")


@pytest.fixture
def mock_llm():
    """Provides a mock LlamaIndex LLM instance."""
    return MagicMock(name="mock_llm")


@pytest.fixture
def mock_query_engine():
    """Provides a mock RetrieverQueryEngine instance."""
    return MagicMock(name="mock_query_engine")


@pytest.fixture
def mock_synthesizer():
    """Provides a mock response synthesizer."""
    return MagicMock(name="mock_synthesizer")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestBasicRAGInitialization:

    @patch("future_rag.pipelines.basic_rag.RetrieverQueryEngine")
    @patch("future_rag.pipelines.basic_rag.get_response_synthesizer")
    def test_init_with_default_prompt(
        self,
        mock_get_synthesizer,
        mock_query_engine,
        mock_retriever,
        mock_llm,
        mock_synthesizer,
    ):
        """Verify that BasicRAG uses the default QA template."""
        # Arrange
        mock_get_synthesizer.return_value = mock_synthesizer
        mock_query_engine.return_value = MagicMock(
            name="mock_query_engine_instance"
        )

        # Act
        rag = BasicRAG(
            retriever=mock_retriever,
            llm=mock_llm,
        )

        # Assert
        mock_get_synthesizer.assert_called_once_with(
            llm=mock_llm,
            text_qa_template=DEFAULT_QA_TEMPLATE,
        )

        mock_query_engine.assert_called_once_with(
            retriever=mock_retriever,
            response_synthesizer=mock_synthesizer,
        )

        assert rag.retriever == mock_retriever
        assert rag.llm == mock_llm
        assert rag.query_engine == mock_query_engine.return_value

    @patch("future_rag.pipelines.basic_rag.RetrieverQueryEngine")
    @patch("future_rag.pipelines.basic_rag.get_response_synthesizer")
    def test_init_with_custom_prompt(
        self,
        mock_get_synthesizer,
        mock_query_engine,
        mock_retriever,
        mock_llm,
        mock_synthesizer,
    ):
        """Verify that BasicRAG uses a custom QA template when provided."""
        # Arrange
        custom_template = MagicMock(
            name="custom_prompt_template"
        )

        mock_get_synthesizer.return_value = mock_synthesizer
        mock_query_engine.return_value = MagicMock(
            name="mock_query_engine_instance"
        )

        # Act
        rag = BasicRAG(
            retriever=mock_retriever,
            llm=mock_llm,
            qa_template=custom_template,
        )

        # Assert
        mock_get_synthesizer.assert_called_once_with(
            llm=mock_llm,
            text_qa_template=custom_template,
        )

        mock_query_engine.assert_called_once_with(
            retriever=mock_retriever,
            response_synthesizer=mock_synthesizer,
        )

        assert rag.retriever == mock_retriever
        assert rag.llm == mock_llm


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestBasicRAGQuery:

    @patch("future_rag.pipelines.basic_rag.RetrieverQueryEngine")
    @patch("future_rag.pipelines.basic_rag.get_response_synthesizer")
    def test_query_delegates_to_query_engine(
        self,
        mock_get_synthesizer,
        mock_query_engine,
        mock_retriever,
        mock_llm,
        mock_synthesizer,
    ):
        """Verify that query() delegates the question to the query engine."""
        # Arrange
        mock_get_synthesizer.return_value = mock_synthesizer

        mock_engine = MagicMock(
            name="mock_query_engine"
        )

        expected_response = MagicMock(
            name="mock_response"
        )

        mock_engine.query.return_value = expected_response
        mock_query_engine.return_value = mock_engine

        rag = BasicRAG(
            retriever=mock_retriever,
            llm=mock_llm,
        )

        question = "What is trustworthy AI?"

        # Act
        result = rag.query(question)

        # Assert
        mock_engine.query.assert_called_once_with(
            question
        )

        assert result == expected_response


# ---------------------------------------------------------------------------
# Async Query
# ---------------------------------------------------------------------------


class TestBasicRAGAQuery:

    @pytest.mark.asyncio
    @patch("future_rag.pipelines.basic_rag.RetrieverQueryEngine")
    @patch("future_rag.pipelines.basic_rag.get_response_synthesizer")
    async def test_aquery_delegates_to_query_engine(
        self,
        mock_get_synthesizer,
        mock_query_engine,
        mock_retriever,
        mock_llm,
        mock_synthesizer,
    ):
        """Verify that aquery() delegates asynchronously to the query engine."""
        # Arrange
        mock_get_synthesizer.return_value = mock_synthesizer

        mock_engine = MagicMock(
            name="mock_query_engine"
        )

        expected_response = MagicMock(
            name="mock_async_response"
        )

        mock_engine.aquery = AsyncMock(
            return_value=expected_response
        )

        mock_query_engine.return_value = mock_engine

        rag = BasicRAG(
            retriever=mock_retriever,
            llm=mock_llm,
        )

        question = "What are the principles of FUTURE-AI?"

        # Act
        result = await rag.aquery(question)

        # Assert
        mock_engine.aquery.assert_awaited_once_with(
            question
        )

        assert result == expected_response


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


class TestBasicRAGRetrieve:

    def test_retrieve_delegates_to_retriever(
        self,
        mock_retriever,
        mock_llm,
    ):
        """Verify that retrieve() delegates to the Retriever."""
        # Arrange
        expected_nodes = [
            MagicMock(name="node_1"),
            MagicMock(name="node_2"),
        ]

        mock_retriever.retrieve.return_value = expected_nodes

        with patch(
            "future_rag.pipelines.basic_rag.get_response_synthesizer"
        ) as mock_get_synthesizer, patch(
            "future_rag.pipelines.basic_rag.RetrieverQueryEngine"
        ) as mock_query_engine:

            mock_get_synthesizer.return_value = MagicMock()
            mock_query_engine.return_value = MagicMock()

            rag = BasicRAG(
                retriever=mock_retriever,
                llm=mock_llm,
            )

            question = "What is AI governance?"

            # Act
            results = rag.retrieve(question)

        # Assert
        mock_retriever.retrieve.assert_called_once_with(
            question
        )

        assert results == expected_nodes
        assert len(results) == 2

    def test_retrieve_handles_empty_results(
        self,
        mock_retriever,
        mock_llm,
    ):
        """Verify that retrieve() correctly returns an empty result."""
        # Arrange
        mock_retriever.retrieve.return_value = []

        with patch(
            "future_rag.pipelines.basic_rag.get_response_synthesizer"
        ) as mock_get_synthesizer, patch(
            "future_rag.pipelines.basic_rag.RetrieverQueryEngine"
        ) as mock_query_engine:

            mock_get_synthesizer.return_value = MagicMock()
            mock_query_engine.return_value = MagicMock()

            rag = BasicRAG(
                retriever=mock_retriever,
                llm=mock_llm,
            )

            # Act
            results = rag.retrieve("non-matching question")

        # Assert
        mock_retriever.retrieve.assert_called_once_with(
            "non-matching question"
        )

        assert results == []