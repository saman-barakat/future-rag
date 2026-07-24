"""
Tests for the index builder (FUTURE-RAG).

These are integration tests: they exercise the real loader, chunker,
embedding model, and ChromaDB-backed vector store together, verifying that
IndexBuilder wires those components together correctly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from llama_index.core import VectorStoreIndex

from future_rag.indexing.embeddings import EmbeddingFactory
from future_rag.indexing.indexer import IndexBuilder
from future_rag.indexing.vectorstore import VectorStore
from future_rag.knowledge.chunker import DocumentChunker
from future_rag.knowledge.loaders import DocumentLoader

TEST_DATA = Path(__file__).parent.parent / "data" / "future-ai.pdf"

pytestmark = pytest.mark.skipif(
    not TEST_DATA.exists(),
    reason=f"Test fixture PDF not found at {TEST_DATA}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def documents():
    """Load the sample PDF once per module; every test uses the same source."""
    return DocumentLoader.load(str(TEST_DATA))


@pytest.fixture(scope="module")
def nodes(documents):
    """Default-chunked nodes, shared by every test that doesn't need a custom chunk size."""
    return DocumentChunker().split(documents)


@pytest.fixture(scope="module")
def embedding():
    """Build the embedding model once per module.

    If network access or credentials are not available, skip the module.
    """
    try:
        return EmbeddingFactory.create()
    except Exception as exc:
        pytest.skip(f"EmbeddingFactory.create() unavailable: {exc}")


@pytest.fixture
def vector_store(tmp_path, request):
    """A connected ChromaVectorStore instance, isolated in a tmp_path directory."""
    db_path = tmp_path / "chroma"

    # ChromaDB collection names must be 3-63 chars and contain only [a-zA-Z0-9_-]
    raw_name = request.node.name
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)[:63]
    if len(clean_name) < 3:
        clean_name = f"test_{clean_name}"

    store = VectorStore(db_path=db_path, collection_name=clean_name)
    return store.connect()


# ---------------------------------------------------------------------------
# build() Tests
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_returns_vector_store_index(self, nodes, embedding, vector_store):
        builder = IndexBuilder(
            embedding_model=embedding, vector_store=vector_store
        )

        index = builder.build(nodes)

        assert index is not None
        assert isinstance(index, VectorStoreIndex)

    def test_indexes_all_nodes(self, nodes, embedding, vector_store):
            builder = IndexBuilder(
                embedding_model=embedding, vector_store=vector_store
            )

            index = builder.build(nodes)

            # Query ChromaDB directly to verify all nodes were inserted
            assert vector_store._collection.count() == len(nodes)


class TestBuildIndexWithEmptyNodes:
    """Building an index from an empty node list shouldn't crash."""

    def test_returns_vector_store_index(self, embedding, vector_store):
        builder = IndexBuilder(
            embedding_model=embedding, vector_store=vector_store
        )

        index = builder.build([])

        assert isinstance(index, VectorStoreIndex)

    def test_retriever_returns_no_results(self, embedding, vector_store):
        builder = IndexBuilder(
            embedding_model=embedding, vector_store=vector_store
        )

        index = builder.build([])
        retriever = index.as_retriever()

        assert retriever.retrieve("anything") == []


class TestIndexRetriever:
    def test_index_can_create_retriever(self, nodes, embedding, vector_store):
        builder = IndexBuilder(
            embedding_model=embedding, vector_store=vector_store
        )

        index = builder.build(nodes)
        retriever = index.as_retriever()

        assert retriever is not None

    def test_retriever_returns_relevant_results(
        self, nodes, embedding, vector_store
    ):
        builder = IndexBuilder(
            embedding_model=embedding, vector_store=vector_store
        )

        index = builder.build(nodes)
        retriever = index.as_retriever(similarity_top_k=2)

        results = retriever.retrieve("future of AI")

        assert len(results) > 0
        assert all(hasattr(result, "node") for result in results)


class TestIndexWithCustomChunkSize:
    def test_builds_successfully(self, documents, embedding, vector_store):
        custom_nodes = DocumentChunker(
            chunk_size=1024, chunk_overlap=256
        ).split(documents)
        builder = IndexBuilder(
            embedding_model=embedding, vector_store=vector_store
        )

        index = builder.build(custom_nodes)

        assert isinstance(index, VectorStoreIndex)

    def test_chunk_size_argument_is_respected(self, documents, nodes):
        """Guards against chunk_size/chunk_overlap silently being ignored."""
        custom_nodes = DocumentChunker(
            chunk_size=1024, chunk_overlap=256
        ).split(documents)

        assert (
            len(custom_nodes) != len(nodes)
            or custom_nodes[0].text != nodes[0].text
        )