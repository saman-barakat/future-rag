"""
Unit tests for the RAGService class (FUTURE-RAG).

These tests isolate RAGService by mocking the RAG pipeline.
No real LLM, retriever, vector database, or RAG pipeline is used.
"""

from unittest.mock import MagicMock

import pytest

from future_rag.services.rag_service import RAGService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline():
    """Provides a mock RAG pipeline."""
    return MagicMock(name="mock_pipeline")


@pytest.fixture
def rag_service(mock_pipeline):
    """Provides a RAGService using the mocked pipeline."""
    return RAGService(
        pipeline=mock_pipeline
    )


@pytest.fixture
def mock_response():
    """Provides a mock LlamaIndex response."""
    response = MagicMock(name="mock_response")
    response.response = "This is the generated answer."

    return response


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestRAGServiceInitialization:

    def test_init_stores_pipeline(
        self,
        mock_pipeline,
    ):
        """Verify that the pipeline is stored during initialization."""
        # Act
        service = RAGService(
            pipeline=mock_pipeline
        )

        # Assert
        assert service.pipeline == mock_pipeline


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestRAGServiceQuery:

    def test_query_delegates_to_pipeline(
        self,
        rag_service,
        mock_pipeline,
        mock_response,
    ):
        """Verify that query() delegates the question to the pipeline."""
        # Arrange
        mock_pipeline.query.return_value = mock_response

        question = "What is trustworthy AI?"

        # Act
        result = rag_service.query(question)

        # Assert
        mock_pipeline.query.assert_called_once_with(
            question
        )

        assert result == "This is the generated answer."

    def test_query_returns_response_text(
        self,
        rag_service,
        mock_pipeline,
    ):
        """Verify that query() returns response.response."""
        # Arrange
        mock_response = MagicMock(
            name="mock_response"
        )
        mock_response.response = (
            "FUTURE-AI provides guidelines for trustworthy AI."
        )

        mock_pipeline.query.return_value = mock_response

        # Act
        result = rag_service.query(
            "What does FUTURE-AI provide?"
        )

        # Assert
        assert result == (
            "FUTURE-AI provides guidelines for trustworthy AI."
        )

    def test_query_rejects_empty_question(
        self,
        rag_service,
        mock_pipeline,
    ):
        """Verify that an empty question raises ValueError."""
        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Question cannot be empty",
        ):
            rag_service.query("")

        # Pipeline should never be called.
        mock_pipeline.query.assert_not_called()

    def test_query_rejects_whitespace_question(
        self,
        rag_service,
        mock_pipeline,
    ):
        """Verify that a whitespace-only question raises ValueError."""
        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Question cannot be empty",
        ):
            rag_service.query("   ")

        # Pipeline should never be called.
        mock_pipeline.query.assert_not_called()

    @pytest.mark.parametrize(
        "question",
        [
            "",
            " ",
            "  ",
            "\t",
            "\n",
            " \t\n ",
        ],
    )
    def test_query_rejects_blank_questions(
        self,
        rag_service,
        mock_pipeline,
        question,
    ):
        """Verify that all blank question variations are rejected."""
        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Question cannot be empty",
        ):
            rag_service.query(question)

        mock_pipeline.query.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline Response
# ---------------------------------------------------------------------------


class TestRAGServiceResponse:

    def test_query_returns_only_response_text(
        self,
        rag_service,
        mock_pipeline,
    ):
        """
        Verify that RAGService returns response.response rather than
        the complete LlamaIndex response object.
        """
        # Arrange
        mock_response = MagicMock(
            name="mock_response"
        )

        mock_response.response = "Generated answer"

        # Add other attributes to demonstrate that they are not returned.
        mock_response.source_nodes = [
            MagicMock(name="source_node")
        ]

        mock_pipeline.query.return_value = mock_response

        # Act
        result = rag_service.query(
            "What is FUTURE-AI?"
        )

        # Assert
        assert result == "Generated answer"
        assert result != mock_response
        assert result != mock_response.source_nodes


# ---------------------------------------------------------------------------
# Pipeline Errors
# ---------------------------------------------------------------------------


class TestRAGServicePipelineErrors:

    def test_query_propagates_pipeline_exception(
        self,
        rag_service,
        mock_pipeline,
    ):
        """Verify that exceptions from the pipeline are propagated."""
        # Arrange
        mock_pipeline.query.side_effect = RuntimeError(
            "RAG pipeline failed."
        )

        # Act & Assert
        with pytest.raises(
            RuntimeError,
            match="RAG pipeline failed",
        ):
            rag_service.query(
                "What is trustworthy AI?"
            )

        mock_pipeline.query.assert_called_once_with(
            "What is trustworthy AI?"
        )