from pathlib import Path

from future_rag.knowledge.loaders import DocumentLoader
from future_rag.knowledge.chunker import DocumentChunker


TEST_DATA = Path(__file__).parent.parent / "data"


def test_chunk_documents():
    """Test splitting documents into chunks."""

    documents = DocumentLoader.load(
        TEST_DATA / "future-ai.pdf"
    )

    chunker = DocumentChunker(
        chunk_size=512,
        chunk_overlap=100,
    )

    nodes = chunker.split(documents)

    assert len(nodes) > 0

    assert all(
        len(node.text.strip()) > 0
        for node in nodes
    )


def test_custom_chunk_size():
    """Smaller chunk sizes should generally produce more chunks."""

    documents = DocumentLoader.load(
        TEST_DATA / "future-ai.pdf"
    )

    small_chunker = DocumentChunker(
        chunk_size=256,
        chunk_overlap=50,
    )

    large_chunker = DocumentChunker(
        chunk_size=1024,
        chunk_overlap=100,
    )

    small_nodes = small_chunker.split(documents)
    large_nodes = large_chunker.split(documents)

    assert len(small_nodes) >= len(large_nodes)


def test_chunk_metadata():
    """Chunk metadata should be preserved."""

    documents = DocumentLoader.load(
        TEST_DATA / "future-ai.pdf"
    )

    chunker = DocumentChunker()

    nodes = chunker.split(documents)

    assert "file_name" in nodes[0].metadata

    assert nodes[0].metadata["file_name"] == "future-ai.pdf"

def test_chunker_configuration():
    """Chunker stores the provided configuration."""

    chunker = DocumentChunker(
        chunk_size=1024,
        chunk_overlap=256,
    )

    assert chunker.splitter.chunk_size == 1024
    assert chunker.splitter.chunk_overlap == 256