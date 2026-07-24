"""
Tests for embedding models.
"""

import pytest

from future_rag.indexing.embeddings import EmbeddingFactory


@pytest.fixture(scope="session")
def embedding():
    """Create the default embedding model once for all tests."""

    return EmbeddingFactory.create()


def test_create_default_embedding(embedding):
    """Test creating the default embedding model."""

    assert embedding is not None


def test_create_custom_embedding():
    """Test creating a custom embedding model."""

    custom_embedding = EmbeddingFactory.create(
        model_name="BAAI/bge-small-en-v1.5"
    )

    assert custom_embedding is not None


def test_generate_embedding(embedding):
    """Test generating an embedding vector."""

    vector = embedding.get_text_embedding(
        "FUTURE-AI is a trustworthy AI guideline."
    )

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_same_text_same_embedding(embedding):
    """The same text should always produce the same embedding."""

    vector1 = embedding.get_text_embedding(
        "Artificial Intelligence"
    )

    vector2 = embedding.get_text_embedding(
        "Artificial Intelligence"
    )

    assert vector1 == vector2


def test_different_text_different_embedding(embedding):
    """Different texts should produce different embeddings."""

    vector1 = embedding.get_text_embedding(
        "Artificial Intelligence"
    )

    vector2 = embedding.get_text_embedding(
        "Software Engineering"
    )

    assert vector1 != vector2


def test_invalid_model():
    """Creating an invalid model should raise an exception."""

    with pytest.raises(Exception):
        EmbeddingFactory.create(
            model_name="invalid/model"
        )